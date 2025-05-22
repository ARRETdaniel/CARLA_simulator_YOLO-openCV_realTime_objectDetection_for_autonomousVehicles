import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
from performance_metrics import PerformanceMetrics

def show_image(img):
    cv.imshow("Image", img)
    cv.waitKey(0)

def draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels):
    # If there are any detections
    if len(idxs) > 0:
        for i in idxs.flatten():
            # Get the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            # Get the unique color for this class
            color = [int(c) for c in colors[classids[i]]]

            # Draw the bounding box rectangle and label on the image
            cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
            text = "{}: {:4f}".format(labels[classids[i]], confidences[i])
            cv.putText(img, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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

#ef infer_image(net, layer_names, height, width, img, colors, labels, FLAGS,
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

def display_object_warnings(frame, boxes, confidences, classids, idxs, metrics=None):
    """
    Displays warning messages for various detected objects with severity based on
    estimated proximity to the ego vehicle.

    Args:
        frame: The input frame where warnings will be displayed
        boxes: Detected bounding boxes
        confidences: Confidence scores for each detection
        classids: Class IDs for each detection
        idxs: Valid detection indices after NMS
        metrics: Optional metrics object to record warning statistics

    Returns:
        frame: The frame with warning messages added
    """
    # Dictionary to store warning metrics
    warning_metrics = {
        'count': 0,
        'types': {},
        'severities': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    }

    # Check if we have valid detections
    if idxs is not None and len(idxs) > 0:
        # Dictionary to store detection information for each object type
        # Key: class_id, Value: (box, confidence, name)
        object_detections = {}

        # Objects of interest with their class IDs from COCO dataset
        objects_of_interest = {
            0: "person",       # Person
            1: "bicycle",      # Bicycle
            2: "car",          # Car
            3: "motorcycle",   # Motorcycle
            5: "bus",          # Bus
            6: "train",        # Train
            7: "truck",        # Truck
            9: "traffic light",# Traffic light
            10: "fire hydrant",# Fire hydrant
            11: "stop sign",   # Stop sign
            12: "parking meter",# Parking meter
            13: "bench"        # Bench
        }

        # Frame dimensions
        height, width = frame.shape[:2]
        frame_area = width * height

        # Check for each object of interest in our detections
        for i in idxs.flatten():
            class_id = classids[i]

            # Skip if not in our objects of interest
            if class_id not in objects_of_interest:
                continue

            # Get object information
            box = boxes[i]
            confidence = confidences[i]
            object_name = objects_of_interest[class_id]

            # Store the detection with highest confidence if multiple of same class
            if class_id not in object_detections or confidence > object_detections[class_id][1]:
                object_detections[class_id] = (box, confidence, object_name)

        # Process and display warnings for detected objects
        warning_y_position = height - 50  # Starting position for warnings
        warning_spacing = 40  # Spacing between warnings

        # Draw warnings for each detected object type
        for class_id, (box, confidence, object_name) in object_detections.items():
            # Calculate relative size as a distance heuristic
            box_area = box[2] * box[3]  # width * height
            relative_size = box_area / frame_area

            # Determine warning level based on relative size
            if relative_size > 0.15:
                severity = "HIGH"
                warning_color = (0, 0, 255)  # Red (BGR)
                warning_prefix = "IMMEDIATE ACTION: "
            elif relative_size > 0.05:
                severity = "MEDIUM"
                warning_color = (0, 165, 255)  # Orange (BGR)
                warning_prefix = "CAUTION: "
            else:
                severity = "LOW"
                warning_color = (0, 255, 255)  # Yellow (BGR)
                warning_prefix = "NOTICE: "

            # Update warning metrics
            warning_metrics['count'] += 1
            warning_metrics['severities'][severity] += 1

            if object_name in warning_metrics['types']:
                warning_metrics['types'][object_name] += 1
            else:
                warning_metrics['types'][object_name] = 1

            # Create warning message
            warning_message = f"{warning_prefix}{object_name.upper()} DETECTED"

            # Special handling for specific objects
            if class_id == 11:  # Stop sign
                if severity == "HIGH":
                    warning_message = "STOP IMMEDIATELY!"
                elif severity == "MEDIUM":
                    warning_message = "PREPARE TO STOP"
                else:
                    warning_message = "Approaching Stop Sign"
            elif class_id == 9:  # Traffic light
                warning_message = f"{warning_prefix}TRAFFIC LIGHT"
            elif class_id == 2 or class_id == 7:  # Car or truck
                if severity == "HIGH":
                    warning_message = f"DANGER: {object_name.upper()} VERY CLOSE"
                else:
                    warning_message = f"{warning_prefix}{object_name.upper()} AHEAD"
            elif class_id == 0:  # Person
                if severity == "HIGH":
                    warning_message = "PEDESTRIAN - IMMEDIATE STOP!"
                else:
                    warning_message = f"{warning_prefix}PEDESTRIAN DETECTED"

            # Draw warning text with background for better visibility
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2

            # Calculate text size
            text_size = cv.getTextSize(warning_message, font, font_scale, thickness)[0]
            text_x = (width - text_size[0]) // 2

            # Draw background rectangle
            cv.rectangle(frame,
                        (text_x - 10, warning_y_position - text_size[1] - 5),
                        (text_x + text_size[0] + 10, warning_y_position + 5),
                        (0, 0, 0),
                        -1)  # Filled rectangle

            # Draw text
            cv.putText(frame, warning_message, (text_x, warning_y_position), font,
                      font_scale, warning_color, thickness)

            # Move position for next warning
            warning_y_position -= warning_spacing

    # Record warning metrics if metrics object is provided
    if metrics:
        metrics.record_warning_metrics(warning_metrics)

    return frame
