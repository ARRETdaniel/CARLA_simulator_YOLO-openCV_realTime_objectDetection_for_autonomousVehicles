import time
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from datetime import datetime

class PerformanceMetrics:
    def __init__(self, output_dir="metrics_output"):
        """Initialize the performance metrics collector.

        Args:
            output_dir (str): Directory to save metrics data and visualizations
        """
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Detection metrics
        self.detection_times = []
        self.detection_counts = []
        self.detection_classes = {}
        self.confidence_scores = []

        # Warning metrics
        self.warnings_generated = []
        self.warning_types = {}
        self.warning_severities = {
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0
        }

        # Timestamps
        self.timestamps = []

        # Init timestamp
        self.init_time = datetime.now()
        self.last_frame_time = time.time()

        # Create log files
        self.detection_log_file = os.path.join(output_dir, "detection_metrics.csv")
        self.warning_log_file = os.path.join(output_dir, "warning_metrics.csv")

        # Initialize log files with headers
        with open(self.detection_log_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'frame_time', 'detection_time', 'num_detections',
                             'class_distribution', 'avg_confidence'])

        with open(self.warning_log_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'warnings_count', 'warning_types',
                            'high_severity', 'medium_severity', 'low_severity'])

    def record_detection_metrics(self, detection_time, boxes, confidences, classids, idxs):
        """Record metrics related to object detection.

        Args:
            detection_time (float): Time taken for detection in seconds
            boxes (list): Bounding boxes of detected objects
            confidences (list): Confidence scores for detections
            classids (list): Class IDs for detected objects
            idxs (ndarray or tuple): Valid detection indices after NMS
        """
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time

        # Record detection time
        self.detection_times.append(detection_time)

        # Count detections
        num_detections = len(idxs) if idxs is not None else 0
        self.detection_counts.append(num_detections)

        # Record timestamp
        timestamp = (datetime.now() - self.init_time).total_seconds()
        self.timestamps.append(timestamp)

        # Record class distribution
        class_dist = {}
        if idxs is not None:
            # Verifica se idxs é um array NumPy ou uma tupla
            if hasattr(idxs, 'flatten'):
                # É um array NumPy, podemos usar flatten()
                idx_list = idxs.flatten()
            else:
                # É uma tupla ou outro tipo, converter em lista
                idx_list = list(idxs)

            for i in idx_list:
                class_id = classids[i]
                if class_id in class_dist:
                    class_dist[class_id] += 1
                else:
                    class_dist[class_id] = 1

                # Update global class distribution
                if class_id in self.detection_classes:
                    self.detection_classes[class_id] += 1
                else:
                    self.detection_classes[class_id] = 1

        # Calculate average confidence
        avg_confidence = 0
        if idxs is not None and len(idxs) > 0:
            # Também precisamos adaptar aqui
            if hasattr(idxs, 'flatten'):
                idx_list = idxs.flatten()
            else:
                idx_list = list(idxs)

            conf_sum = sum([confidences[i] for i in idx_list])
            avg_confidence = conf_sum / len(idxs)
            self.confidence_scores.append(avg_confidence)

        # Log to file
        with open(self.detection_log_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, frame_time, detection_time, num_detections,
                             str(class_dist), avg_confidence])

    def record_warning_metrics(self, warnings_data):
        """Record metrics related to warning messages.

        Args:
            warnings_data (dict): Dictionary containing warning information
                {
                    'count': Number of warnings displayed,
                    'types': {'stop_sign': 1, 'car': 2, ...},
                    'severities': {'HIGH': 1, 'MEDIUM': 1, 'LOW': 1}
                }
        """
        timestamp = (datetime.now() - self.init_time).total_seconds()

        # Update warning counts
        self.warnings_generated.append(warnings_data['count'])

        # Update warning types
        for warning_type, count in warnings_data['types'].items():
            if warning_type in self.warning_types:
                self.warning_types[warning_type] += count
            else:
                self.warning_types[warning_type] = count

        # Update severity counts
        for severity, count in warnings_data['severities'].items():
            self.warning_severities[severity] += count

        # Log to file
        with open(self.warning_log_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, warnings_data['count'],
                            str(warnings_data['types']),
                            warnings_data['severities'].get('HIGH', 0),
                            warnings_data['severities'].get('MEDIUM', 0),
                            warnings_data['severities'].get('LOW', 0)])

    def generate_summary(self):
        """Generate and save summary statistics from the collected metrics.

        Returns:
            dict: Summary statistics
        """
        summary = {
            'avg_detection_time': np.mean(self.detection_times) if self.detection_times else 0,
            'max_detection_time': np.max(self.detection_times) if self.detection_times else 0,
            'min_detection_time': np.min(self.detection_times) if self.detection_times else 0,
            'avg_fps': 1.0 / (np.mean(self.detection_times) if self.detection_times and np.mean(self.detection_times) > 0 else 1),
            'total_detections': np.sum(self.detection_counts) if self.detection_counts else 0,
            'avg_detections_per_frame': np.mean(self.detection_counts) if self.detection_counts else 0,
            'detection_classes_distribution': self.detection_classes,
            'avg_confidence': np.mean(self.confidence_scores) if self.confidence_scores else 0,
            'total_warnings': np.sum(self.warnings_generated) if self.warnings_generated else 0,
            'warning_types_distribution': self.warning_types,
            'warning_severities': self.warning_severities,
            'total_runtime_seconds': (datetime.now() - self.init_time).total_seconds()
        }

        # Save summary to file
        with open(os.path.join(self.output_dir, 'summary_stats.csv'), 'w') as f:
            for key, value in summary.items():
                f.write(f"{key},{value}\n")

        return summary

    def visualize_metrics(self):
        """Generate and save visualizations of the performance metrics."""
        if not self.timestamps:
            print("No data to visualize")
            return

        # Create figure with subplots
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))

        # 1. Detection Time Over Time
        axs[0, 0].plot(self.timestamps, self.detection_times)
        axs[0, 0].set_title('Detection Time (s) vs Time')
        axs[0, 0].set_xlabel('Time (s)')
        axs[0, 0].set_ylabel('Detection Time (s)')
        axs[0, 0].grid(True)

        # 2. Number of Detections Over Time
        axs[0, 1].plot(self.timestamps, self.detection_counts)
        axs[0, 1].set_title('Number of Detections vs Time')
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('Count')
        axs[0, 1].grid(True)

        # 3. Class Distribution Pie Chart
        if self.detection_classes:
            labels = [f"Class {cls}" for cls in self.detection_classes.keys()]
            sizes = list(self.detection_classes.values())
            axs[1, 0].pie(sizes, labels=labels, autopct='%1.1f%%')
            axs[1, 0].set_title('Object Class Distribution')

        # 4. Average Confidence Over Time
        if self.confidence_scores:
            # Garantir que temos o mesmo número de timestamps e scores
            min_len = min(len(self.timestamps), len(self.confidence_scores))
            axs[1, 1].plot(self.timestamps[:min_len], self.confidence_scores[:min_len])
            axs[1, 1].set_title('Average Confidence vs Time')
            axs[1, 1].set_xlabel('Time (s)')
            axs[1, 1].set_ylabel('Confidence')
            axs[1, 1].grid(True)

        # 5. Warnings Over Time
        if self.warnings_generated:
            # Criar timestamps específicos para warnings se necessário
            if len(self.timestamps) >= len(self.warnings_generated):
                # Usar os timestamps existentes
                warning_timestamps = self.timestamps[:len(self.warnings_generated)]
            else:
                # Criar timestamps interpolados para warnings
                warning_timestamps = np.linspace(
                    self.timestamps[0],
                    self.timestamps[-1],
                    len(self.warnings_generated)
                )

            axs[2, 0].plot(warning_timestamps, self.warnings_generated)
            axs[2, 0].set_title('Warnings Generated vs Time')
            axs[2, 0].set_xlabel('Time (s)')
            axs[2, 0].set_ylabel('Count')
            axs[2, 0].grid(True)

        # 6. Warning Severity Distribution
        if sum(self.warning_severities.values()) > 0:
            labels = list(self.warning_severities.keys())
            sizes = list(self.warning_severities.values())
            axs[2, 1].bar(labels, sizes)
            axs[2, 1].set_title('Warning Severity Distribution')
            axs[2, 1].set_xlabel('Severity')
            axs[2, 1].set_ylabel('Count')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_metrics.png'))
        plt.close()

        # Create a histogram of detection times
        plt.figure(figsize=(10, 6))
        plt.hist(self.detection_times, bins=30)
        plt.title('Distribution of Detection Times')
        plt.xlabel('Detection Time (s)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'detection_times_histogram.png'))
        plt.close()
