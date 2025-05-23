import numpy as np
import cv2
import time
import os
import threading
import subprocess
from collections import deque

# Global warning timers with persistence
warning_timers = {
    'person': 0,
    'car': 0,
    'truck': 0,
    'bus': 0,
    'stop sign': 0,
    'traffic light': 0,
    'bicycle': 0,
    'motorcycle': 0
}

# Persistence duration for each warning type
WARNING_PERSISTENCE = {
    'person': 15,       # Critical - keep warnings for 15 frames
    'stop sign': 15,    # Critical - keep warnings for 15 frames
    'traffic light': 10,# Important - keep warnings for 10 frames
    'default': 8        # Standard - keep warnings for 8 frames
}

# Pre-computed warning overlays (initialized later)
WARNING_OVERLAYS = {}

# Last audio warning timestamps
last_audio_warning = {}

class OptimizedYOLO:
    def __init__(self, model_type="tiny", input_size=(320, 320),
                 confidence_threshold=0.5, nms_threshold=0.3,
                 use_opencl=True):
        """Initialize the optimized YOLO detector

        Args:
            model_type: "tiny" for YOLOv3-tiny, "full" for YOLOv3
            input_size: Network input size (smaller = faster)
            confidence_threshold: Detection confidence threshold
            nms_threshold: Non-maximum suppression threshold
            use_opencl: Whether to use OpenCL acceleration
        """
        self.prev_indices = []
        self.model_type = model_type
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        # Load configuration based on model type
        #if model_type == "tiny":
        if False:
            self.config_path = './yolov3-coco/yolov3-tiny.cfg'
            self.weights_path = './yolov3-coco/yolov3-tiny.weights'
        else:
            self.config_path = './yolov3-coco/yolov3.cfg'
            self.weights_path = './yolov3-coco/yolov3.weights'

        self.labels_path = './yolov3-coco/coco-labels'

        # Check if files exist
        self._check_files()

        # Load labels
        self.labels = open(self.labels_path).read().strip().split('\n')

        # Create colors
        np.random.seed(42)  # For reproducible colors
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype='uint8')

        # Load neural network
        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)

        # Set backend
        if use_opencl:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
            print("Using OpenCL acceleration")
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("Using CPU for inference")

        # Get output layer names
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i-1] for i in self.net.getUnconnectedOutLayers()]

        # Initialize warning overlays
        self._initialize_warning_overlays()

        # Create audio executor thread pool (single thread to avoid spawning many threads)
        self.audio_thread = None

        # For frame skipping
        self.skip_frame = False

        # Store previous detections for persistence
        self.prev_boxes = []
        self.prev_confidences = []
        self.prev_classids = []

        # Detection buffer for reusing in _process_detections
        self.detection_buffer = {
            'boxes': [],
            'confidences': [],
            'classids': []
        }

        # Warm-up the network with a dummy image
        self._warmup()

        # Debugging and performance monitoring
        self.debug = True  # Enable/disable detailed timing info
        self.fps_history = deque(maxlen=30)  # Store FPS history for smoothing

    def _check_files(self):
        """Check if model files exist"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Weights file not found: {self.weights_path}")
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")

    def _warmup(self):
        """Perform a warmup inference to initialize the network"""
        dummy_image = np.zeros((416, 416, 3), dtype=np.uint8)
        self.detect(dummy_image, warmup=True)
        print("Network warm-up complete")

    def _initialize_warning_overlays(self):
        """Pre-compute warning overlays for different numbers of warnings"""
        global WARNING_OVERLAYS

        # Create warning overlays for 1-3 warnings
        for num_warnings in range(1, 4):
            height = min(num_warnings, 3) * 45 + 15

            # Create multiple sizes to handle different image resolutions
            for width in [800, 640, 320]:
                # Create base overlay (black background with transparency)
                overlay = np.zeros((height, width, 3), dtype=np.uint8)
                WARNING_OVERLAYS[(num_warnings, width)] = overlay

    def detect(self, img, metrics=None, warmup=False):
        """Run detection on an image

        Args:
            img: Input image
            metrics: Optional metrics object for performance tracking
            warmup: Whether this is a warmup call (don't process results)

        Returns:
            img: Processed image with detections
            boxes: Bounding boxes
            confidences: Confidence scores
            classids: Class IDs
            idxs: Valid detection indices after NMS
        """
        # Get image dimensions
        height, width = img.shape[:2]

        # Frame skipping for higher FPS (process every other frame)
        if self.skip_frame and not warmup:
            self.skip_frame = False
            # Return the previous results
            img = self._draw_results(img, self.prev_boxes, self.prev_confidences,
                                    self.prev_classids, self.prev_indices)
            return img, self.prev_boxes, self.prev_confidences, self.prev_classids, self.prev_indices
        else:
            self.skip_frame = True

        # Start timing
        start_time = time.time()

        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(img, 1/255.0, self.input_size,
                                    swapRB=True, crop=False)

        # Set input to the network
        self.net.setInput(blob)

        # Forward pass
        preprocess_time = time.time() - start_time
        inference_start = time.time()
        outputs = self.net.forward(self.output_layers)
        inference_time = time.time() - inference_start

        postprocess_start = time.time()
        boxes, confidences, classids = self._process_detections(outputs, width, height)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        postprocess_time = time.time() - postprocess_start
        # Measure detection time
        detection_time = time.time() - start_time

        # Store for frame skipping
        self.prev_boxes = boxes
        self.prev_confidences = confidences
        self.prev_classids = classids
        self.prev_indices = indices

        # Record metrics if provided
        if metrics and not warmup:
            metrics.record_detection_metrics(detection_time, boxes, confidences,
                                           classids, indices)

        # Skip further processing if this is just a warmup
        if warmup:
            return img, boxes, confidences, classids, indices

        # Record FPS for smoother reporting
        self.fps_history.append(1/detection_time)
        avg_fps = sum(self.fps_history)/len(self.fps_history)

        if self.debug:
            # Create a cleaner, more readable performance report
            performance_str = (
                f"\n{'=' * 35}\n"
                f"YOLO Performance Metrics:\n"
                f"  ⏱️ Preprocessing: {preprocess_time*1000:.1f}ms\n"
                f"  🧠 Inference:     {inference_time*1000:.1f}ms\n"
                f"  🔍 Postprocess:   {postprocess_time*1000:.1f}ms\n"
                f"  📊 Total time:    {detection_time*1000:.1f}ms\n"
                f"  🚀 Current FPS:   {1/detection_time:.1f}\n"
                f"  📈 Average FPS:   {avg_fps:.1f}\n"
                f"  🎯 Detections:    {len(indices) if isinstance(indices, tuple) else (len(indices.flatten()) if len(indices) > 0 else 0)}\n"
                f"{'=' * 35}"
            )
            print(performance_str)
        else:
            # Simple performance summary for production
            print(f"[INFO] YOLOv3-{'tiny' if self.model_type == 'tiny' else 'full'} @ {self.input_size}: {avg_fps:.1f} FPS")

        # Draw results on the image
        img = self._draw_results(img, boxes, confidences, classids, indices)

        return img, boxes, confidences, classids, indices

    def _process_detections(self, outputs, width, height):
        """Process network outputs to get boxes, confidences, and class IDs"""
        max_detections = sum(len(output) for output in outputs)
        boxes = []
        confidences = []
        classids = []
        boxes_capacity = max(100, max_detections)  # Reasonable initial capacity

        # Process each output layer
        for output in outputs:
            for detection in output:
                # Get class ID and confidence
                scores = detection[5:]
                classid = np.argmax(scores)
                confidence = scores[classid]

                # Filter by confidence threshold
                if confidence > self.confidence_threshold:
                    # Scale the bounding box coordinates to the image size
                    box = detection[0:4] * np.array([width, height, width, height])
                    center_x, center_y, box_width, box_height = box.astype('int')

                    # Calculate top-left corner coordinates
                    x = int(center_x - (box_width / 2))
                    y = int(center_y - (box_height / 2))

                    # Add to lists
                    boxes.append([x, y, int(box_width), int(box_height)])
                    confidences.append(float(confidence))
                    classids.append(classid)

        # Reuse the detection results
        self.detection_buffer['boxes'] = boxes
        self.detection_buffer['confidences'] = confidences
        self.detection_buffer['classids'] = classids

        return boxes, confidences, classids

    def _draw_results(self, img, boxes, confidences, classids, indices):
        """Draw detection results and warnings on the image"""
        # Priority classes for warnings
        priority_classes = {
            'person': 'CAUTION: PEDESTRIAN DETECTED',
            'car': 'VEHICLE AHEAD',
            'truck': 'LARGE VEHICLE AHEAD',
            'bus': 'BUS AHEAD',
            'stop sign': 'APPROACHING STOP SIGN',
            'traffic light': 'TRAFFIC LIGHT AHEAD',
            'bicycle': 'CYCLIST NEARBY',
            'motorcycle': 'MOTORCYCLE NEARBY'
        }

        # Get image dimensions
        height, width = img.shape[:2]

        # Tracking for warnings
        critical_warnings = []
        standard_warnings = []
        detected_classes = set()

        # Ensure indices is a valid array
        if isinstance(indices, tuple):
            # For older versions of OpenCV
            idx_list = indices
        elif len(indices) > 0:
            # For newer versions of OpenCV that return a 2D array
            idx_list = indices.flatten()
        else:
            idx_list = []

        # Draw boxes and collect warnings
        for i in idx_list:
            # Get box coordinates
            x, y, w, h = boxes[i]

            # Get the label and color
            class_id = classids[i]
            if class_id < len(self.labels):
                label = self.labels[class_id]
                color = [int(c) for c in self.colors[class_id]]

                # Draw the box and label
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                text = f"{label}: {confidences[i]:.2f}"
                cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Check if this is a class we want to warn about
                if label in priority_classes:
                    detected_classes.add(label)
                    warning_msg = priority_classes[label]
                    box_area = w * h  # Proxy for distance
                    conf_percent = int(confidences[i] * 100)

                    # Add to appropriate warning list
                    if label in ['person', 'stop sign']:
                        critical_warnings.append((warning_msg, conf_percent, box_area, label))
                        # Play audio warning
                        self._play_audio_warning(label)
                    else:
                        standard_warnings.append((warning_msg, conf_percent, box_area, label))

        # Update warning timers for detected classes
        for class_name in detected_classes:
            persistence = WARNING_PERSISTENCE.get(class_name, WARNING_PERSISTENCE['default'])
            warning_timers[class_name] = persistence

        # Handle warning persistence for classes not detected in this frame
        for class_name, timer in list(warning_timers.items()):
            if class_name not in detected_classes and timer > 0:
                # Decrease timer
                warning_timers[class_name] = timer - 1

                # Add persisted warning with reduced confidence
                if timer > 2:  # Only show persisted warnings that are still relevant
                    fading_conf = max(40, int(70 * (timer / WARNING_PERSISTENCE.get(class_name, WARNING_PERSISTENCE['default']))))
                    warning_msg = priority_classes.get(class_name, "Warning")
                    size_factor = 0.75  # Smaller than direct detections

                    # Add to appropriate warning list with "(Persisted)" tag
                    if class_name in ['person', 'stop sign']:
                        critical_warnings.append((f"{warning_msg} (Persisted)", fading_conf, size_factor, class_name))
                    else:
                        standard_warnings.append((f"{warning_msg} (Persisted)", fading_conf, size_factor, class_name))

        # Display warnings if we have any
        if critical_warnings or standard_warnings:
            # Sort by importance (box size/proximity)
            critical_warnings.sort(key=lambda x: x[2], reverse=True)
            standard_warnings.sort(key=lambda x: x[2], reverse=True)

            # Combine warnings, prioritizing critical ones
            all_warnings = critical_warnings[:2] + standard_warnings[:1]

            if all_warnings:
                # Get the pre-computed overlay or create one if needed
                num_warnings = min(len(all_warnings), 3)

                # Find the closest width match
                if width >= 640:
                    overlay_width = 800
                elif width >= 320:
                    overlay_width = 640
                else:
                    overlay_width = 320

                # Get the overlay
                key = (num_warnings, overlay_width)
                if key not in WARNING_OVERLAYS:
                    # Create new overlay if needed (should rarely happen)
                    warning_height = num_warnings * 45 + 15
                    overlay = np.zeros((warning_height, overlay_width, 3), dtype=np.uint8)
                    WARNING_OVERLAYS[key] = overlay

                # Create a copy of the base overlay
                overlay = WARNING_OVERLAYS[key].copy()
                warning_height = overlay.shape[0]

                # Create a semi-transparent black background
                overlay_region = img[height - warning_height:height, 0:width].copy()
                cv2.addWeighted(overlay[:, :width], 0.7, overlay_region, 0.3, 0, overlay_region)
                img[height - warning_height:height, 0:width] = overlay_region

                # Display warnings
                y_offset = height - warning_height + 40
                for warning, conf, _, class_name in all_warnings[:3]:
                    # Choose icon and color based on warning type
                    if "PEDESTRIAN" in warning:
                        icon = "!"
                        text_color = (0, 0, 255)  # Red
                        warning_with_icon = f"{icon} {warning}"
                    elif "STOP SIGN" in warning:
                        icon = "!"
                        text_color = (0, 0, 255)  # Red
                        warning_with_icon = f"{icon} {warning}"
                    elif "VEHICLE" in warning or "TRUCK" in warning or "BUS" in warning:
                        icon = ">"
                        text_color = (255, 255, 0)  # Yellow
                        warning_with_icon = f"{icon} {warning}"
                    elif "CYCLIST" in warning or "MOTORCYCLE" in warning:
                        icon = ">"
                        text_color = (255, 255, 0)  # Yellow
                        warning_with_icon = f"{icon} {warning}"
                    elif "TRAFFIC LIGHT" in warning:
                        icon = ">"
                        text_color = (0, 255, 255)  # Light blue
                        warning_with_icon = f"{icon} {warning}"
                    else:
                        icon = "!"
                        text_color = (255, 255, 255)  # White
                        warning_with_icon = f"{icon} {warning}"

                    # Draw text with shadow for better visibility
                    cv2.putText(img, f"{warning_with_icon} ({conf}%)",
                              (22, y_offset+2), cv2.FONT_HERSHEY_SIMPLEX,
                              0.75, (0, 0, 0), 4)  # Shadow

                    cv2.putText(img, f"{warning_with_icon} ({conf}%)",
                              (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                              0.75, text_color, 2)

                    # Add separator line
                    if y_offset + 45 < height:
                        cv2.line(img, (30, y_offset + 15), (width - 30, y_offset + 15),
                               (200, 200, 200), 1)

                    y_offset += 45

        return img

    def _play_audio_warning(self, warning_type):
        """Play audio warning for critical detections"""
        # Check if we should play a warning (avoid too frequent warnings)
        current_time = time.time()
        last_time = last_audio_warning.get(warning_type, 0)

        # Only play every 3 seconds for each type
        if current_time - last_time < 3.0:
            return

        # Update last warning time
        last_audio_warning[warning_type] = current_time

        # Don't create a new thread if one is still running
        if self.audio_thread and self.audio_thread.is_alive():
            return

        # Choose sound based on warning type
        if warning_type == 'person':
            sound_command = "powershell -c (New-Object Media.SoundPlayer 'C:\\Windows\\Media\\Windows Exclamation.wav').PlaySync();"
        elif warning_type == 'stop sign':
            sound_command = "powershell -c (New-Object Media.SoundPlayer 'C:\\Windows\\Media\\Windows Critical Stop.wav').PlaySync();"
        else:
            sound_command = "powershell -c (New-Object Media.SoundPlayer 'C:\\Windows\\Media\\Windows Notify.wav').PlaySync();"

        # Create and start thread
        try:
            self.audio_thread = threading.Thread(
                target=lambda: subprocess.run(sound_command, shell=True, timeout=2))
            self.audio_thread.daemon = True
            self.audio_thread.start()
        except Exception as e:
            print(f"Warning: Audio playback error: {e}")


# Function to infer image compatible with original code
def infer_image_optimized(yolo, img, metrics=None):
    """Process image with optimized YOLO detector

    Args:
        yolo: OptimizedYOLO instance
        img: Input image
        metrics: Optional metrics object

    Returns:
        img: Processed image
        boxes: Bounding boxes
        confidences: Confidence scores
        classids: Class IDs
        idxs: Valid indices
    """
    return yolo.detect(img, metrics=metrics)
