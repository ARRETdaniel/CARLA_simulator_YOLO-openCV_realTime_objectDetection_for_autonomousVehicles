Running the CARLA simulator
============================

Before running the CARLA simulator, ensure you have installed all the required dependencies. You can do this by checking the `requirements.txt` file and installing the necessary packages:

```bash
pip install -r requirements.txt
```
============================

In one terminal, start the CARLA simulator at a 30hz fixed time-step:

Ubuntu:
-------
```bash
./CarlaUE4.sh /Game/Maps/Course4 -windowed -carla-server -benchmark -fps=30
```

Windows:
--------
```bash
CarlaUE4.exe /Game/Maps/Course4 -windowed -carla-server -benchmark -fps=30
```

Running the Python client (and controller)
==========================================

In another terminal, change the directory to go into the "Course4FinalProject" folder, under the "PythonClient" folder.

Run your controller, execute the following command while CARLA is open:

Ubuntu (use alternative python commands if the command below does not work, as described in the CARLA install guide):
------
```bash
python3 module_7.py
```

Windows (use alternative python commands if the command below does not work, as described in the CARLA install guide):
--------
```bash
python module_7.py
```


CarlaUE4.exe /Game/Maps/Course4 -windowed -carla-server -benchmark -fps=30 CarlaUE4.exe /Game/Maps/Course4 -quality-level=Low -windowed -carla-server -benchmark -fps=30 CarlaUE4.exe /Game/Maps/Course4 -RenderOffScreen -windowed -carla-server -benchmark -fps=30 CarlaUE4.exe /Game/Maps/Course4 -quality-level=Epic -windowed -carla-server -benchmark -fps=30 CarlaUE4.exe -windowed -carla-server -benchmark -fps=30

CarlaUE4.exe /Game/Maps/Town02 -windowed -carla-server -benchmark -fps=60

python module_7.py

linux

./CarlaUE4.sh /Game/Maps/Course4 -windowed -carla-server -benchmark -fps=30
python3.6 module_7.py
python3 module_7.py

darknet detector test C:/src/darknet/cfg/coco.data C:/src/darknet/cfg/yolov7.cfg C:/src/darknet/yolov7.weights C:/Users/danie/Documents/Documents/CURSOS/Self-Driving_Cars_Specialization/CarlaSimulator/PythonClient/Course4FinalProject/_out/episode_3360/CameraRGB/000001.png




wsl --shutdown


linux docker

Rebuild the Docker image: After modifying the Dockerfile, rebuild the Docker image:
docker build -t carla_simulator_image .


Run your Docker container again:
docker run -it --rm carla_simulator_image


docker run -it --rm carla_simulator_image bash
exit
