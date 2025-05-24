# detector_client.py (to be imported in module_7.py)
import socket
import time
import numpy as np
import msgpack
import msgpack_numpy as m
import cv2
import colorsys
from collections import deque
import threading
import subprocess
m.patch()  # Enable numpy array serialization

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

# Pre-computed warning overlays
WARNING_OVERLAYS = {}

# Last audio warning timestamps
last_audio_warning = {}

class DetectionClient:
    def __init__(self, host="localhost", port=5555, reconnect_attempts=3):
        self.host = host
        self.port = port
        self.socket = None
        self.reconnect_attempts = reconnect_attempts
        self.connected = False
        self.labels = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        # Initialize FPS history tracker
        self.fps_history = deque(maxlen=30)  # Store FPS history for smoothing

        # Initialize warning overlays system
        self._initialize_warning_overlays()

        # Audio thread
        self.audio_thread = None

        # Connect to server
        self.connect()

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

    def connect(self):
        """Connect to the detection server"""
        attempts = 0
        while attempts < self.reconnect_attempts:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.host, self.port))
                self.connected = True
                print(f"Connected to detection server at {self.host}:{self.port}")
                break
            except socket.error as e:
                attempts += 1
                print(f"Connection attempt {attempts} failed: {e}")
                time.sleep(1)

        if not self.connected:
            print("Failed to connect to detection server")

    def detect_objects(self, image_array):
        """Send image to server and get detection results"""
        if not self.connected:
            self.connect()
            if not self.connected:
                return image_array, [], [], [], []

        try:
            # Pack image data
            data = {'image': image_array}
            packed_data = msgpack.packb(data, default=m.encode)

            # Send data size followed by data
            self.socket.sendall(len(packed_data).to_bytes(4, byteorder='little'))
            self.socket.sendall(packed_data)

            # Receive result size
            result_size_bytes = self.socket.recv(4)
            if not result_size_bytes:
                raise ConnectionError("No data received from server")

            result_size = int.from_bytes(result_size_bytes, byteorder='little')

            # Receive results
            received_data = b''
            while len(received_data) < result_size:
                chunk = self.socket.recv(min(4096, result_size - len(received_data)))
                if not chunk:
                    break
                received_data += chunk

            # Unpack results
            results = msgpack.unpackb(received_data, raw=False, object_hook=m.decode)

            boxes = results['boxes']
            confidences = results['confidences']
            class_ids = results['class_ids']
            fps = results.get('fps', 0)
            processing_time = results.get('processing_time', 0)

            # Add to FPS history for smoothing
            self.fps_history.append(fps)
            avg_fps = sum(self.fps_history) / len(self.fps_history)

            # Print detailed performance metrics
            self._print_performance_metrics(results)

            # Draw bounding boxes on a copy of the image
            processed_img = np.copy(image_array)

            # Track detected classes for warnings
            detected_classes = set()

            # Draw boxes and identify warnings
            for i, box in enumerate(boxes):
                x, y, w, h = box
                conf = confidences[i]
                cls_id = class_ids[i]
                label = self.labels[cls_id] if cls_id < len(self.labels) else "unknown"

                # Generate color based on class ID
                hue = cls_id / len(self.labels)
                color = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.7, 1.0)]

                # Track detected classes for warnings
                if label in ['person', 'car', 'truck', 'bus', 'stop sign',
                             'traffic light', 'bicycle', 'motorcycle']:
                    detected_classes.add(label)

                # Draw bounding box
                processed_img = cv2.rectangle(processed_img, (x, y),
                                            (x + w, y + h), color, 2)

                # Draw label with confidence
                text = f"{label}: {conf:.2f}"
                processed_img = cv2.putText(processed_img, text, (x, y - 10),
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Add warnings based on detections
            processed_img = self._add_warnings(processed_img, boxes, confidences, class_ids, detected_classes)

            # Add FPS info
            processed_img = cv2.putText(processed_img, f"YOLOv8 Detection: {avg_fps:.1f} FPS",
                                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                     0.7, (0, 255, 0), 2)

            # Return processed image and detection results
            indices = np.arange(len(boxes))  # Simple indices for compatibility
            return processed_img, boxes, confidences, class_ids, indices

        except Exception as e:
            print(f"Error communicating with detection server: {e}")
            self.connected = False
            return image_array, [], [], [], []

    def _add_warnings(self, img, boxes, confidences, class_ids, detected_classes):
        """Add warning overlays to the image based on detections"""
        global warning_timers

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

        # Process each detection
        for i, cls_id in enumerate(class_ids):
            if cls_id < len(self.labels):
                label = self.labels[cls_id]
                conf = confidences[i]
                x, y, w, h = boxes[i]

                # Check if this is a class we want to warn about
                if label in priority_classes:
                    warning_msg = priority_classes[label]
                    box_area = w * h  # Proxy for distance
                    conf_percent = int(conf * 100)

                    # Add to appropriate warning list
                    if label in ['person', 'stop sign']:
                        critical_warnings.append((warning_msg, conf_percent, box_area, label))
                        # Play audio warning
                        self._play_audio_warning(label)
                    else:
                        standard_warnings.append((warning_msg, conf_percent, box_area, label))

        # Update warning timers for detected classes
        for class_name in detected_classes:
            if class_name in warning_timers:
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
                    # Create new overlay if needed
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
                    elif "STOP SIGN" in warning:
                        icon = "!"
                        text_color = (0, 0, 255)  # Red
                    elif "VEHICLE" in warning or "TRUCK" in warning or "BUS" in warning:
                        icon = ">"
                        text_color = (255, 255, 0)  # Yellow
                    elif "CYCLIST" in warning or "MOTORCYCLE" in warning:
                        icon = ">"
                        text_color = (255, 255, 0)  # Yellow
                    elif "TRAFFIC LIGHT" in warning:
                        icon = ">"
                        text_color = (0, 255, 255)  # Light blue
                    else:
                        icon = "!"
                        text_color = (255, 255, 255)  # White

                    warning_with_icon = f"{warning} ({conf}%)"

                    # Draw text with shadow for better visibility
                    cv2.putText(img, f"{warning_with_icon}",
                              (22, y_offset+2), cv2.FONT_HERSHEY_SIMPLEX,
                              0.75, (0, 0, 0), 4)  # Shadow

                    cv2.putText(img, f"{warning_with_icon}",
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
        global last_audio_warning

        # Check if we should play a warning (avoid too frequent warnings)
        current_time = time.time()
        last_time = last_audio_warning.get(warning_type, 0)

        # Only play every 3 seconds for each type
        if current_time - last_time < 3.0:
            return

        # Update last warning time
        last_audio_warning[warning_type] = current_time

        # Don't create a new thread if one is still running
        if hasattr(self, 'audio_thread') and self.audio_thread and self.audio_thread.is_alive():
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

    def _print_performance_metrics(self, results):
        """Print detailed performance metrics about the detection"""
        # Skip if no performance data available
        if 'processing_time' not in results:
            return

        processing_time = results['processing_time']
        fps = results.get('fps', 1.0/processing_time if processing_time > 0 else 0)
        inference_time = results.get('inference_time', processing_time * 0.7)  # Estimate if not provided
        preprocess_time = results.get('preprocess_time', processing_time * 0.1)
        postprocess_time = results.get('postprocess_time', processing_time * 0.2)
        model_type = results.get('model_type', 'YOLOv8')

        # Calculate average FPS
        avg_fps = sum(self.fps_history) / len(self.fps_history)

        # Create a cleaner, more readable performance report
        performance_str = (
            f"\n{'=' * 35}\n"
            f"{model_type} Detection Metrics:\n"
            f"  ⏱️ Preprocessing: {preprocess_time*1000:.1f}ms\n"
            f"  🧠 Inference:     {inference_time*1000:.1f}ms\n"
            f"  🔍 Postprocess:   {postprocess_time*1000:.1f}ms\n"
            f"  📊 Total time:    {processing_time*1000:.1f}ms\n"
            f"  🚀 Current FPS:   {fps:.1f}\n"
            f"  📈 Average FPS:   {avg_fps:.1f}\n"
            f"  🎯 Detections:    {len(results['boxes'])}\n"
            f"{'=' * 35}"
        )
        print(performance_str)

    def close(self):
        """Close the connection"""
        if self.socket:
            self.socket.close()
            self.socket = None
            self.connected = False
