import time
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import json

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

        # NEW: Detection accuracy metrics by class
        self.class_true_positives = {}  # {class_id: count}
        self.class_false_negatives = {}  # {class_id: count}
        self.class_confidence_by_id = {}  # {class_id: [confidence scores]}

        # NEW: Detection by distance metrics
        self.distance_bands = {
            'próximo': {'detections': 0, 'total_objects': 0},  # Objetos próximos (menos de 10m estimados)
            'médio': {'detections': 0, 'total_objects': 0},    # Objetos a média distância (10-30m)
            'distante': {'detections': 0, 'total_objects': 0}  # Objetos distantes (>30m)
        }

        # Timestamps
        self.timestamps = []

        # Init timestamp
        self.init_time = datetime.now()
        self.last_frame_time = time.time()

        # Create log files
        self.detection_log_file = os.path.join(output_dir, "detection_metrics.csv")
        self.warning_log_file = os.path.join(output_dir, "warning_metrics.csv")
        self.class_accuracy_file = os.path.join(output_dir, "class_accuracy.csv")
        self.distance_metrics_file = os.path.join(output_dir, "distance_metrics.csv")

        # Initialize log files with headers
        with open(self.detection_log_file, 'w', encoding='latin-1') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'frame_time', 'detection_time', 'num_detections',
                             'class_distribution', 'avg_confidence'])

        with open(self.warning_log_file, 'w', encoding='latin-1') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'warnings_count', 'warning_types',
                            'high_severity', 'medium_severity', 'low_severity'])

        with open(self.class_accuracy_file, 'w', encoding='latin-1') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'class_id', 'true_positives', 'false_negatives',
                             'precision', 'avg_confidence'])

        with open(self.distance_metrics_file, 'w', encoding='latin-1') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'distance_band', 'detections', 'total_objects', 'detection_rate'])

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

                # NEW: Update class confidence scores
                if class_id not in self.class_confidence_by_id:
                    self.class_confidence_by_id[class_id] = []
                self.class_confidence_by_id[class_id].append(confidences[i])

                # NEW: Update true positives by class (assuming all detections are true positives)
                # This is a simplification - ideally would be validated against ground truth
                if class_id not in self.class_true_positives:
                    self.class_true_positives[class_id] = 0
                self.class_true_positives[class_id] += 1

            # NEW: Calculate distance metrics based on bounding box size
            if boxes and len(boxes) > 0:
                frame_area = 800 * 600  # Assuming standard frame size, adjust if different
                for i in idx_list:
                    box = boxes[i]
                    box_area = box[2] * box[3]  # width * height
                    relative_size = box_area / frame_area

                    # Estimate distance band based on relative size
                    if relative_size > 0.1:
                        self.distance_bands['próximo']['detections'] += 1
                        self.distance_bands['próximo']['total_objects'] += 1
                    elif relative_size > 0.02:
                        self.distance_bands['médio']['detections'] += 1
                        self.distance_bands['médio']['total_objects'] += 1
                    else:
                        self.distance_bands['distante']['detections'] += 1
                        self.distance_bands['distante']['total_objects'] += 1

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

        # NEW: Log class accuracy (simplified - assuming all detections are correct)
        for class_id in class_dist:
            precision = 1.0  # Simplified - would need ground truth to calculate real precision

            # Calculate average confidence for this class in this frame
            class_confidences = [confidences[i] for i in idx_list if classids[i] == class_id]
            avg_class_confidence = sum(class_confidences) / len(class_confidences) if class_confidences else 0

            with open(self.class_accuracy_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    class_id,
                    class_dist[class_id],  # True positives
                    0,                     # False negatives (simplified - would need ground truth)
                    precision,
                    avg_class_confidence
                ])

        # NEW: Log distance metrics
        for band, data in self.distance_bands.items():
            detection_rate = data['detections'] / max(data['total_objects'], 1)
            with open(self.distance_metrics_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    band,
                    data['detections'],
                    data['total_objects'],
                    detection_rate
                ])

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
            'detection_classes_distribution': {str(k): v for k, v in self.detection_classes.items()},
            'avg_confidence': np.mean(self.confidence_scores) if self.confidence_scores else 0,
            'total_warnings': np.sum(self.warnings_generated) if self.warnings_generated else 0,
            'warning_types_distribution': self.warning_types,
            'warning_severities': self.warning_severities,
            'total_runtime_seconds': (datetime.now() - self.init_time).total_seconds(),
            # Convert numerical class keys to strings for the class precision and confidence dictionaries
            'class_precision': {str(cls): self.class_true_positives.get(cls, 0) / max(self.class_true_positives.get(cls, 0) + self.class_false_negatives.get(cls, 0), 1)
                               for cls in set(list(self.class_true_positives.keys()) + list(self.class_false_negatives.keys()))},
            'class_avg_confidence': {str(cls): np.mean(scores) if scores else 0
                                    for cls, scores in self.class_confidence_by_id.items()},
            'distance_detection_rates': {band: data['detections'] / max(data['total_objects'], 1)
                                       for band, data in self.distance_bands.items()}
        }

        # Save summary to file
        with open(os.path.join(self.output_dir, 'summary_stats.csv'), 'w') as f:
            for key, value in summary.items():
                # Convert dictionaries to JSON strings for better storage
                if isinstance(value, dict):
                    f.write(f"{key},{json.dumps(value)}\n")
                else:
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
        axs[0, 0].set_title('Tempo de Detecção (s) vs Tempo')
        axs[0, 0].set_xlabel('Tempo (s)')
        axs[0, 0].set_ylabel('Tempo de Detecção (s)')
        axs[0, 0].grid(True)

        # 2. Number of Detections Over Time
        axs[0, 1].plot(self.timestamps, self.detection_counts)
        axs[0, 1].set_title('Número de Detecções vs Tempo')
        axs[0, 1].set_xlabel('Tempo (s)')
        axs[0, 1].set_ylabel('Contagem')
        axs[0, 1].grid(True)

        # 3. Class Distribution Pie Chart
        if self.detection_classes:
            labels = [self._get_class_name(cls) for cls in self.detection_classes.keys()]
            sizes = list(self.detection_classes.values())
            axs[1, 0].pie(sizes, labels=labels, autopct='%1.1f%%')
            axs[1, 0].set_title('Distribuição de Classes de Objetos')

        # 4. Average Confidence Over Time
        if self.confidence_scores:
            # Garantir que temos o mesmo número de timestamps e scores
            min_len = min(len(self.timestamps), len(self.confidence_scores))
            axs[1, 1].plot(self.timestamps[:min_len], self.confidence_scores[:min_len])
            axs[1, 1].set_title('Confiança Média vs Tempo')
            axs[1, 1].set_xlabel('Tempo (s)')
            axs[1, 1].set_ylabel('Confiança')
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
            axs[2, 0].set_title('Avisos Gerados vs Tempo')
            axs[2, 0].set_xlabel('Tempo (s)')
            axs[2, 0].set_ylabel('Contagem')
            axs[2, 0].grid(True)

        # 6. Warning Severity Distribution
        if sum(self.warning_severities.values()) > 0:
            severity_translation = {
                "HIGH": "ALTA",
                "MEDIUM": "MÉDIA",
                "LOW": "BAIXA"
            }
            labels = [severity_translation.get(sev, sev) for sev in self.warning_severities.keys()]
            sizes = list(self.warning_severities.values())
            axs[2, 1].bar(labels, sizes)
            axs[2, 1].set_title('Distribuição de Severidade dos Avisos')
            axs[2, 1].set_xlabel('Severidade')
            axs[2, 1].set_ylabel('Contagem')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_metrics.png'))
        plt.close()

        # Create a histogram of detection times
        plt.figure(figsize=(10, 6))
        plt.hist(self.detection_times, bins=30)
        plt.title('Distribuição dos Tempos de Detecção')
        plt.xlabel('Tempo de Detecção (s)')
        plt.ylabel('Frequência')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'detection_times_histogram.png'))
        plt.close()

        # NEW: Class precision chart
        if self.class_true_positives:
            plt.figure(figsize=(10, 6))
            class_ids = sorted(self.class_true_positives.keys())
            precisions = [self.class_true_positives.get(cls_id, 0) / max(self.class_true_positives.get(cls_id, 0) +
                         self.class_false_negatives.get(cls_id, 0), 1) for cls_id in class_ids]
            confidences = [np.mean(self.class_confidence_by_id.get(cls_id, [0])) for cls_id in class_ids]

            # Plot bars for precision and confidence
            x = np.arange(len(class_ids))
            width = 0.35

            plt.bar(x - width/2, precisions, width, label='Precisão')
            plt.bar(x + width/2, confidences, width, label='Confiança Média')

            plt.xlabel('Classes')
            plt.ylabel('Pontuação')
            plt.title('Precisão e Confiança por Classe')
            plt.xticks(x, [self._get_class_name(cls_id) for cls_id in class_ids], rotation=45, ha='right')
            plt.ylim(0, 1.0)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()  # Adicionado para melhorar o layout com rótulos rotacionados
            plt.savefig(os.path.join(self.output_dir, 'class_precision_metrics.png'))
            plt.close()

        # NEW: Detection rate by distance chart
        if sum(data['detections'] for data in self.distance_bands.values()) > 0:
            plt.figure(figsize=(10, 6))
            bands = list(self.distance_bands.keys())
            detection_counts = [data['detections'] for data in self.distance_bands.values()]
            total_objects = [data['total_objects'] for data in self.distance_bands.values()]
            detection_rates = [data['detections'] / max(data['total_objects'], 1)
                              for data in self.distance_bands.values()]

            # Tradução para as faixas de distância
            band_labels = {
                'próximo': 'Próximo',
                'médio': 'Médio',
                'distante': 'Distante'
            }

            # Create a stacked bar for detections vs total objects
            x = np.arange(len(bands))
            width = 0.35

            plt.bar(x, detection_counts, width, label='Detecções')
            missed_objects = [total - detected for total, detected in zip(total_objects, detection_counts)]
            plt.bar(x, missed_objects, width, bottom=detection_counts,
                   label='Objetos Perdidos', alpha=0.5)

            # Add line for detection rate
            ax2 = plt.twinx()
            ax2.plot(x, detection_rates, 'ro-', linewidth=2, label='Taxa de Detecção')
            ax2.set_ylim(0, 1.0)
            ax2.set_ylabel('Taxa de Detecção')

            plt.xlabel('Faixa de Distância')
            plt.ylabel('Contagem')
            plt.title('Desempenho de Detecção por Distância')
            plt.xticks(x, [band_labels.get(band, band) for band in bands])

            # Combine legends from both axes
            lines1, labels1 = plt.gca().get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'distance_metrics.png'))
            plt.close()

    def _get_class_name(self, class_id):
        """Retorna nome da classe em português com base no ID, focado em cenários de direção"""
        class_names = {
            0: "Pedestre",
            1: "Bicicleta",
            2: "Veículo",
            3: "Motocicleta",
            5: "Ônibus",
            6: "Trem",
            7: "Caminhão",
            8: "Veículo aquático",
            9: "Semáforo",
            10: "Hidrante",
            11: "Placa de pare",
            12: "Parquímetro",
            13: "Mobiliário urbano"
        }
        return class_names.get(class_id, f"Objeto {class_id}")
