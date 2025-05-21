from results_reporter import ResultsReporter
import os

def create_dummy_metrics():
    """Create dummy metrics files for testing the reporter"""
    metrics_dir = "test_metrics"

    # Create directory if it doesn't exist
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    # Create summary stats
    with open(os.path.join(metrics_dir, 'summary_stats.csv'), 'w') as f:
        f.write("avg_detection_time,0.23\n")
        f.write("max_detection_time,0.42\n")
        f.write("min_detection_time,0.18\n")
        f.write("avg_fps,4.35\n")
        f.write("total_detections,127\n")
        f.write("avg_detections_per_frame,2.5\n")
        f.write("detection_classes_distribution,{2: 85, 13: 32, 0: 10}\n")
        f.write("avg_confidence,0.72\n")
        f.write("total_warnings,45\n")
        f.write("warning_types_distribution,{\"car\": 30, \"person\": 10, \"bench\": 5}\n")
        f.write("warning_severities,{\"HIGH\": 15, \"MEDIUM\": 20, \"LOW\": 10}\n")
        f.write("total_runtime_seconds,65.4\n")

    # Create detection metrics file
    with open(os.path.join(metrics_dir, 'detection_metrics.csv'), 'w') as f:
        f.write("timestamp,frame_time,detection_time,num_detections,class_distribution,avg_confidence\n")
        for i in range(10):
            f.write(f"{i*2.5},{0.25},{0.21},{2},{{2: 2}},{0.85}\n")

    # Create warning metrics file
    with open(os.path.join(metrics_dir, 'warning_metrics.csv'), 'w') as f:
        f.write("timestamp,warnings_count,warning_types,high_severity,medium_severity,low_severity\n")
        for i in range(10):
            f.write(f"{i*2.5},{2},{{\"car\": 2}},{1},{1},{0}\n")

    return metrics_dir

def test_reporter():
    """Test the ResultsReporter class"""
    # Create dummy metrics files
    metrics_dir = create_dummy_metrics()
    report_dir = "test_report"

    # Create reporter instance
    reporter = ResultsReporter(metrics_dir=metrics_dir, report_dir=report_dir)

    # Generate the report
    reporter.generate_report()

    print(f"Test report generated at {report_dir}/performance_report.html")
    print(f"Open this file in a web browser to view the report.")

if __name__ == "__main__":
    test_reporter()
