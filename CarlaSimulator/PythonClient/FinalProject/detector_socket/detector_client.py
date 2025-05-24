# detector_client.py (to be imported in module_7.py)
import socket
import time
import numpy as np
import msgpack
import msgpack_numpy as m
import cv2
m.patch()  # Enable numpy array serialization

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
        self.connect()

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
                return None, [], [], [], []

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

            # Draw bounding boxes on a copy of the image
            processed_img = np.copy(image_array)

            # Simple visualization for debugging
            for i, box in enumerate(boxes):
                x, y, w, h = box
                label = f"Class {class_ids[i]}: {confidences[i]:.2f}"
                color = (0, 255, 0)  # Green color for boxes

                # Draw bounding box
                processed_img = cv2.rectangle(processed_img, (x, y),
                                            (x + w, y + h), color, 2)

                # Draw label
                processed_img = cv2.putText(processed_img, label, (x, y - 10),
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Add FPS info
            processed_img = cv2.putText(processed_img, f"Detection FPS: {fps:.1f}",
                                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                     1, (0, 255, 0), 2)

            # Return processed image and detection results
            indices = np.arange(len(boxes))  # Simple indices for compatibility
            return processed_img, boxes, confidences, class_ids, indices

        except Exception as e:
            print(f"Error communicating with detection server: {e}")
            self.connected = False
            return image_array, [], [], [], []

    def close(self):
        """Close the connection"""
        if self.socket:
            self.socket.close()
            self.socket = None
            self.connected = False
