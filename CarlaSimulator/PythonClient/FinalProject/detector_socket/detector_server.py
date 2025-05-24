# detector_server.py (Python 3.12)
import socket
import json
import numpy as np
import cv2
from ultralytics import YOLO
import time
import threading
import msgpack
import msgpack_numpy as m
m.patch()  # Patch msgpack to handle numpy arrays efficiently

class DetectionServer:
    def __init__(self, host="localhost", port=5555,
                 model_type="yolov8s.pt", confidence=0.4):
        self.host = host
        self.port = port
        self.confidence = confidence

        # Check CUDA availability before loading model
        import torch
        self.has_cuda = torch.cuda.is_available()

        if self.has_cuda:
            print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
            self.device = "0"  # Use first GPU
        else:
            print("CUDA is not available. Using CPU")
            self.device = "cpu"

        # Initialize YOLO model
        print(f"Loading {model_type} model...")
        self.model = YOLO(model_type)
        # Initialize socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"Detection server running on {host}:{port}")

    def process_image(self, image_data):
        # Process the image with YOLO
        results = self.model(image_data, conf=self.confidence, device=self.device) # Use GPU 0

        # Extract detection results
        boxes = []
        confidences = []
        class_ids = []

        # Process results for the first (and only) image
        result = results[0]

        for box in result.boxes:
            # Get the box coordinates in [x, y, w, h] format for compatibility
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x, y = int(x1), int(y1)
            width, height = int(x2 - x1), int(y2 - y1)

            # Get confidence and class
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            boxes.append([x, y, width, height])
            confidences.append(conf)
            class_ids.append(cls_id)

        # Return the results in the format expected by the CARLA client
        return {
            'boxes': boxes,
            'confidences': confidences,
            'class_ids': class_ids
        }

    def handle_client(self, client_socket):
        """Handle communication with a single client"""
        try:
            # Process images continuously from this client
            while True:
                # First receive the size of the incoming data
                data_size_bytes = client_socket.recv(4)
                if not data_size_bytes:
                    break

                data_size = int.from_bytes(data_size_bytes, byteorder='little')

                # Receive the image data
                received_data = b''
                while len(received_data) < data_size:
                    chunk = client_socket.recv(min(4096, data_size - len(received_data)))
                    if not chunk:
                        break
                    received_data += chunk

                # Unpack the image data
                unpacked_data = msgpack.unpackb(received_data, raw=False, object_hook=m.decode)
                image = unpacked_data['image']

                # Process the image
                start_time = time.time()
                detection_results = self.process_image(image)
                processing_time = time.time() - start_time

                # Add processing time to the results
                detection_results['processing_time'] = processing_time
                detection_results['fps'] = 1.0 / processing_time

                # Pack the results
                packed_results = msgpack.packb(detection_results, default=m.encode)

                # Send the size first, then the data
                result_size = len(packed_results).to_bytes(4, byteorder='little')
                client_socket.sendall(result_size + packed_results)

        except Exception as e:
            print(f"Error processing client data: {e}")
        finally:
            client_socket.close()

    def run(self):
        """Run the detection server"""
        try:
            while True:
                print("Waiting for client connection...")
                client_socket, addr = self.server_socket.accept()
                print(f"Client connected from {addr}")

                # Handle each client in a separate thread
                client_thread = threading.Thread(target=self.handle_client,
                                               args=(client_socket,))
                client_thread.daemon = True
                client_thread.start()

        except KeyboardInterrupt:
            print("Server shutdown requested")
        finally:
            self.server_socket.close()
            print("Server closed")

if __name__ == "__main__":
    # Create and run the detection server
    server = DetectionServer(model_type="yolov8s.pt")
    server.run()
