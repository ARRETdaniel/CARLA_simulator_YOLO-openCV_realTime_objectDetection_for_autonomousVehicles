# Self-Driving Cars

Welcome to my college final project (TCC) on Self-Driving Cars! This repository contains all the materials and code.

## Project Overview

This project was developed as part of the Self-Driving Cars Specialization, which covers various aspects of autonomous vehicle technology, including:

- Computer Vision
- Camara Sensor
- Localization
- Path Planning
- Control

## Getting Started

To get started with the project, follow these steps:

1. Clone the repository:
  ```bash
  git clone https://github.com/ARRETdaniel/CARLA_simulator_YOLO-openCV_realTime_objectDetection_for_autonomousVehicles.git
  ```
2. Download CarlaUE4 modified version [CARLA Installation Guide](https://github.com/ARRETdaniel/CARLA_simulator_YOLO-openCV_realTime_objectDetection_for_autonomousVehicles/tree/main/CARLA%20Installation%20Guide) in the CarlaSimulator folder.

3. Follow the instructions in the CARLA Installation Guide folder for troubleshooting and additional info or/and to set up the environment and run the code.

## Prerequisites

Make sure you have the following software installed (It'll only work in version mentioned below):

- CarlaUE4 modified version
- Python 3.6.x
- Required Python libraries (listed in `CarlaSimulator/PythonClient/FinalProject/requirements.txt`)

## Install Python Dependencies:

```bash
cd CarlaSimulator/PythonClient/FinalProject
pip install -r requirements.txt
```

## Running the CARLA Simulator (Windows)

1. Launch CARLA in Server Mode:

Open a command prompt and execute the following commands to start the CARLA simulator on Town 1 (CarlaUE4 modified version):

```bash
C:
cd \CarlaSimulator\CarlaUE4\Binaries\Win64
CarlaUE4.exe /Game/Maps/Course4 -windowed -carla-server -benchmark -fps=30
```
Note: The -benchmark and -fps=30 flags set a fixed time-step mode, which is essential for consistent simulation results.

## Running the Custom Solution

2. Execute the Python Client:

In another command prompt, run your custom solution script:

```bash
C:
cd \CarlaSimulator\PythonClient\FinalProject
python module_7.py
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

We would like to thank the instructors and contributors of the Self-Driving Cars Specialization for their valuable resources and guidance.

Happy learning and coding!
