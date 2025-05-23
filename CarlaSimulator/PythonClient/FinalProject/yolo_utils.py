import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
import threading
from performance_metrics import PerformanceMetrics

# Global dictionary to track warning persistence
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

# Persistence duration in frames for each warning type
WARNING_PERSISTENCE = {
    'person': 15,       # Critical - keep warnings for 15 frames
    'stop sign': 15,    # Critical - keep warnings for 15 frames
    'traffic light': 10,# Important - keep warnings for 10 frames
    'default': 8        # Standard - keep warnings for 8 frames
}

# Dictionary to track last audio warning time to prevent rapid repeats
last_audio_warning = {
    'person': 0,
    'stop sign': 0
}

# Audio feedback function using a separate thread to avoid blocking
def play_audio_warning(warning_type):
    """Play audio warning without blocking the main detection thread"""
    current_time = time.time()

    # Only play audio warning if it's been at least 3 seconds since the last one of this type
    if current_time - last_audio_warning.get(warning_type, 0) < 3.0:
        return

    # Update the last warning time
    last_audio_warning[warning_type] = current_time

    # Different sounds for different warning types
    if warning_type == 'person':
        sound_command = "powershell -c (New-Object Media.SoundPlayer 'C:\\Windows\\Media\\Windows Exclamation.wav').PlaySync();"
    elif warning_type == 'stop sign':
        sound_command = "powershell -c (New-Object Media.SoundPlayer 'C:\\Windows\\Media\\Windows Critical Stop.wav').PlaySync();"
    else:
        # Default sound for other warnings
        sound_command = "powershell -c (New-Object Media.SoundPlayer 'C:\\Windows\\Media\\Windows Notify.wav').PlaySync();"

    # Run the command in a separate thread to avoid blocking
    threading.Thread(target=lambda: subprocess.run(sound_command, shell=True)).start()

def show_image(img):
    cv.imshow("Image", img)
    cv.waitKey(0)

def draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels):
    # If there are any detections
    if len(idxs) > 0:
        # Priority classes that should trigger warnings
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

        # Track important detections
        critical_warnings = []
        standard_warnings = []

        # Track which classes were detected in this frame
        detected_classes = set()

        for i in idxs.flatten():
            # Get the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            # Get the unique color for this class
            color = [int(c) for c in colors[classids[i]]]

            # Draw the bounding box rectangle and label on the image
            cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
            text = "{}: {:.2f}".format(labels[classids[i]], confidences[i])
            cv.putText(img, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Check if this is a class we want to warn about
            class_name = labels[classids[i]]
            if class_name in priority_classes:
                detected_classes.add(class_name)
                warning_msg = priority_classes[class_name]
                box_area = w * h  # Use as proxy for distance/importance
                conf_percent = int(confidences[i] * 100)

                # Separate critical warnings from standard ones
                if class_name in ['person', 'stop sign']:
                    critical_warnings.append((warning_msg, conf_percent, box_area, class_name))
                    # Trigger audio warning for critical detections
                    play_audio_warning(class_name)
                else:
                    standard_warnings.append((warning_msg, conf_percent, box_area, class_name))

        # Update warning timers: reset timer for detected classes
        for class_name in detected_classes:
            persistence = WARNING_PERSISTENCE.get(class_name, WARNING_PERSISTENCE['default'])
            warning_timers[class_name] = persistence

        # Decrease timers for non-detected classes and include warnings for classes with active timers
        for class_name, timer in list(warning_timers.items()):
            if class_name not in detected_classes and timer > 0:
                # This class wasn't detected in current frame but timer is still active
                warning_timers[class_name] = timer - 1

                # Add a persisted warning with reducing confidence
                fading_conf = max(40, int(70 * (timer / WARNING_PERSISTENCE.get(class_name, WARNING_PERSISTENCE['default']))))
                warning_msg = priority_classes.get(class_name, "Warning")

                # Use a smaller size factor for persistent warnings to indicate they're older
                size_factor = 0.75  # Smaller than direct detections

                if class_name in ['person', 'stop sign']:
                    critical_warnings.append((f"{warning_msg} (Persisted)", fading_conf, size_factor, class_name))
                else:
                    standard_warnings.append((f"{warning_msg} (Persisted)", fading_conf, size_factor, class_name))

        # Display warnings if we have any
        if critical_warnings or standard_warnings:
            # Sort by box size (larger = closer/more important)
            critical_warnings.sort(key=lambda x: x[2], reverse=True)
            standard_warnings.sort(key=lambda x: x[2], reverse=True)

            # Combine warnings, prioritizing critical ones
            all_warnings = critical_warnings[:2] + standard_warnings[:1]

            if all_warnings:
                # Get image dimensions
                height, width = img.shape[:2]

                # Create a more visually appealing background for warnings (gradient)
                overlay = img.copy()
                warning_height = min(len(all_warnings), 3) * 45 + 15

                # Create a gradient overlay (dark at bottom, lighter toward top)
                for i in range(warning_height):
                    alpha = min(0.8, 0.4 + (i / warning_height) * 0.4)  # Gradient transparency
                    cv.rectangle(overlay,
                               (0, height - i),
                               (width, height),
                               (0, 0, 0),
                               -1)
                    cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

                # Display warnings
                y_offset = height - warning_height + 40
                for warning, conf, _, class_name in all_warnings[:3]:
                    # Choose warning icon based on type
                    if "PEDESTRIAN" in warning:
                        icon = "🚶‍"
                        text_color = (0, 0, 255)  # Red for pedestrians
                        warning_with_icon = f"{icon} {warning}"
                    elif "STOP SIGN" in warning:
                        icon = "🛑"
                        text_color = (0, 0, 255)  # Red for stop signs
                        warning_with_icon = f"{icon} {warning}"
                    elif "VEHICLE" in warning or "TRUCK" in warning or "BUS" in warning:
                        icon = "🚗"
                        text_color = (255, 255, 0)  # Yellow for vehicles
                        warning_with_icon = f"{icon} {warning}"
                    elif "CYCLIST" in warning or "MOTORCYCLE" in warning:
                        icon = "🚲"
                        text_color = (255, 255, 0)  # Yellow for cyclists/motorcycles
                        warning_with_icon = f"{icon} {warning}"
                    elif "TRAFFIC LIGHT" in warning:
                        icon = "🚦"
                        text_color = (0, 255, 255)  # Light blue for traffic lights
                        warning_with_icon = f"{icon} {warning}"
                    else:
                        icon = "⚠️"
                        text_color = (255, 255, 255)  # White for others
                        warning_with_icon = f"{icon} {warning}"

                    # Add shadow effect for better visibility
                    cv.putText(img, f"{warning_with_icon} ({conf}%)",
                             (22, y_offset+2), cv.FONT_HERSHEY_SIMPLEX,
                             0.75, (0, 0, 0), 4)  # Shadow

                    # Draw the actual warning text
                    cv.putText(img, f"{warning_with_icon} ({conf}%)",
                             (20, y_offset), cv.FONT_HERSHEY_SIMPLEX,
                             0.75, text_color, 2)

                    # Add a small separation line between warnings
                    if y_offset + 45 < height:
                        cv.line(img, (30, y_offset + 15), (width - 30, y_offset + 15),
                              (200, 200, 200), 1)

                    y_offset += 45

    return img


def generate_boxes_confidences_classids(outs, height, width, tconf):
    boxes = []
    confidences = []
    classids = []

    for out in outs:
        for detection in out:
            #print (detection)
            #a = input('GO!')

            # Get the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]

            # Consider only the predictions that are above a certain confidence level
            if confidence > tconf:
                # TODO Check detection
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, bwidth, bheight = box.astype('int')

                # Using the center x, y coordinates to derive the top
                # and the left corner of the bounding box
                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))

                # Append to list
                boxes.append([x, y, int(bwidth), int(bheight)])
                confidences.append(float(confidence))
                classids.append(classid)

    return boxes, confidences, classids

def infer_image(net, layer_names, height, width, img, colors, labels,
            boxes=None, confidences=None, classids=None, idxs=None, infer=True, metrics=None):
    confidence = 0.5
    threshold = 0.3
    if infer:
        # Medir todo o processo de detecção
        start_time = time.time()
        # Contructing a blob from the input image
        blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                        swapRB=True, crop=False)

        # Perform a forward pass of the YOLO object detector
        net.setInput(blob)
        outs = net.forward(layer_names)

        # Generate the boxes, confidences, and classIDs
        boxes, confidences, classids = generate_boxes_confidences_classids(outs, height, width, confidence)

        # Apply Non-Maxima Suppression
        idxs = cv.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

        detection_time = time.time() - start_time
        print(f"[INFO] YOLOv3 complete pipeline took {detection_time:.6f} seconds")

        # Record metrics if a metrics object is provided
        if metrics:
            metrics.record_detection_metrics(detection_time, boxes, confidences, classids, idxs)

    if boxes is None or confidences is None or idxs is None or classids is None:
        raise '[ERROR] Required variables are set to None before drawing boxes on images.'

    # Draw labels and boxes on the image
    img = draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels)

    return img, boxes, confidences, classids, idxs
