# Date: Jun 30, 2025

"""
CARLA waypoint.

A controller to follow a given trajectory, where the trajectory
can be defined using way-points.

STARTING in a moment...
"""
from __future__ import print_function
from __future__ import division

# System level imports
import sys
import os
import argparse
import logging
import time
import math
import numpy as np
import csv
import matplotlib.pyplot as plt
import controller2d
import configparser
import local_planner
import behavioural_planner
import cv2 # image processig
import colorsys
import atexit
import csv
import json
import time
import glob
import traceback
from datetime import datetime
import seaborn as sns


#test show  image console
from PIL import Image
# Script level imports
sys.path.append(os.path.abspath(sys.path[0] + '/..'))
import live_plotter as lv   # Custom live plotting library
#import CarlaSimulator.PythonClient.live_plotter as lv   # Custom live plotting library
from carla.sensor            import Camera
from carla.client     import make_carla_client, VehicleControl
from carla.settings   import CarlaSettings
from carla.tcp        import TCPConnectionError
from carla.controller import utils
from carla.image_converter import to_rgb_array, to_bgra_array
from PIL import Image

'''TEST CAMERA'''
from carla.util import print_over_same_line

from yolo import YOLO, infer_image_optimized
#from yolo_utils import infer_image, show_image
from performance_metrics import PerformanceMetrics
from results_reporter import ResultsReporter
from detector_socket.detector_client import DetectionClient
from threaded_detector import ThreadedDetector

## darknet imports
#from pexpect import popen_spawn

# OPEN CV2 WITH DARKNET WEIGHTS ETC.

'''
# Give the configuration, weight and labels files for the model
model_configuration = '.\yolov3-coco\yolov3.cfg';
model_weights = '.\yolov3-coco\yolov3.weights';
model_labels = '.\yolov3-coco\coco-labels';

# Get the labels
labels = open(model_labels).read().strip().split('\n')
# Intializing colors to represent each label uniquely
#colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
# Load the weights and configutation to form the pretrained YOLOv3 model
net = cv2.dnn.readNetFromDarknet(model_configuration, model_weights)
# Get the output layer names of the model
layer_names = net.getLayerNames()
layer_names = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
#layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
'''

''' BEGORE SOCKET
# Initialize the optimized YOLO detector
# Using tiny model with 320x320 input for performance
yolo_detector = YOLO(
    model_type="v3",           # Using tiny model for speed
    input_size=(416, 416),       # Smaller input resolution for better performance
#    confidence_threshold=0.5,    # Only detections above 50% confidence
#    nms_threshold=0.3,          # NMS threshold
    use_opencl=False             # Use OpenCL acceleration
)
'''

''' BEFORE SOCKET
'''
# Generate colors for visualization
np.random.seed(42)  # For reproducible colors
# Get the labels from the YOLO detector



"""
Configurable params
"""
ITER_FOR_SIM_TIMESTEP  = 10     # no. iterations to compute approx sim timestep
WAIT_TIME_BEFORE_START = 1.00   # game seconds (time before controller start)
TOTAL_RUN_TIME         = 100.00 # game seconds (total runtime before sim end)
TOTAL_FRAME_BUFFER     = 300    # number of frames to buffer after total runtime
NUM_PEDESTRIANS        = 0      # total number of pedestrians to spawn
NUM_VEHICLES           = 2      # total number of vehicles to spawn
SEED_PEDESTRIANS       = 0      # seed for pedestrian spawn randomizer
SEED_VEHICLES          = 0      # seed for vehicle spawn randomizer
CLIENT_WAIT_TIME       = 3      # wait time for client before starting episode
                                # used to make sure the server loads
                                # consistently

WEATHERID = {
    "DEFAULT": 0,
    "CLEARNOON": 1,
    "CLOUDYNOON": 2,
    "WETNOON": 3,
    "WETCLOUDYNOON": 4,
    "MIDRAINYNOON": 5,
    "HARDRAINNOON": 6,
    "SOFTRAINNOON": 7,
    "CLEARSUNSET": 8,
    "CLOUDYSUNSET": 9,
    "WETSUNSET": 10,
    "WETCLOUDYSUNSET": 11,
    "MIDRAINSUNSET": 12,
    "HARDRAINSUNSET": 13,
    "SOFTRAINSUNSET": 14,
}

#SIMWEATHER = WEATHERID["HARDRAINNOON"]     # set simulation weather
SIMWEATHER = WEATHERID["HARDRAINSUNSET"]     # set simulation weather
#SIMWEATHER = WEATHERID["CLEARNOON"]     # set simulation weather

# change the start position. The Carla UE must be open with
#  no parameters related to the map.
PLAYER_START_INDEX = 1      # spawn index for player (keep to 1)
FIGSIZE_X_INCHES   = 8      # x figure size of feedback in inches
FIGSIZE_Y_INCHES   = 8      # y figure size of feedback in inches
PLOT_LEFT          = 0.1    # in fractions of figure width and height
PLOT_BOT           = 0.1
PLOT_WIDTH         = 0.8
PLOT_HEIGHT        = 0.8

WAYPOINTS_FILENAME = 'waypoints.txt'  # waypoint file to load
DIST_THRESHOLD_TO_LAST_WAYPOINT = 2.0  # some distance from last position before
                                       # simulation ends

# Planning Constants
NUM_PATHS = 7
BP_LOOKAHEAD_BASE      = 8.0              # m
BP_LOOKAHEAD_TIME      = 2.0              # s
PATH_OFFSET            = 1.5              # m
CIRCLE_OFFSETS         = [-1.0, 1.0, 3.0] # m, just one circle
CIRCLE_RADII           = [1.5, 1.5, 1.5]  # m
TIME_GAP               = 1.0              # s
PATH_SELECT_WEIGHT     = 10
A_MAX                  = 1.5              # m/s^2
SLOW_SPEED             = 2.0              # m/s
STOP_LINE_BUFFER       = 3.5              # m
LEAD_VEHICLE_LOOKAHEAD = 20.0             # m
LP_FREQUENCY_DIVISOR   = 2                # Frequency divisor to make the
                                          # local planner operate at a lower
                                          # frequency than the controller
                                          # (which operates at the simulation
                                          # frequency). Must be a natural
                                          # number.

# Course 4 specific parameters
C4_STOP_SIGN_FILE        = 'stop_sign_params.txt'
C4_STOP_SIGN_FENCELENGTH = 5        # m
C4_PARKED_CAR_FILE       = 'parked_vehicle_params.txt'

# Path interpolation parameters
INTERP_MAX_POINTS_PLOT    = 10   # number of points used for displaying
                                 # selected path
INTERP_DISTANCE_RES       = 0.01 # distance between interpolated points

# controller output directory
CONTROLLER_OUTPUT_FOLDER = os.path.dirname(os.path.realpath(__file__)) +\
                           '/controller_output/'
# CAMERAS
#720p
#WINDOW_WIDTH = 1280
#WINDOW_HEIGHT = 720
# Half HD
#WINDOW_WIDTH = 960
#WINDOW_HEIGHT = 540
#WINDOW_WIDTH = 800
#WINDOW_HEIGHT = 600
# YOLO's native input size
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
# Mini window size for depth and semantic segmentation cameras
MINI_WINDOW_WIDTH = 320
MINI_WINDOW_HEIGHT = 180


# Add this at the top of the file, after other imports

# Check and install required packages
def check_and_install_dependencies():
    import importlib
    import subprocess
    import sys

    required_packages = ['statsmodels', 'seaborn']

    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package} is already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} has been installed")

# Run dependency check at startup



def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need.
    """
    settings = CarlaSettings()

    # There is no need for non-agent info requests if there are no pedestrians
    # or vehicles.
    get_non_player_agents_info = False
    if (NUM_PEDESTRIANS > 0 or NUM_VEHICLES > 0):
        get_non_player_agents_info = True

    ''' TEST camera '''
    # Now we want to add a couple of cameras to the player vehicle.
    # We will collect the images produced by these cameras every
    # frame.
    # The default camera captures RGB images of the scene.
    camera0 = Camera('CAMERA')
    # Set image resolution in pixels.
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    #camera0.set_image_size(800, 600)
    # Set its position relative to the car in meters.
    camera0.set_position(2.0, 0.0, 1.4)
    camera0.set_rotation(0.0, 0.0, 0.0)
    #camera0.set_position(0.30, 0, 1.30)
    settings.add_sensor(camera0)
    # Let's add another camera producing ground-truth depth.
    camera1 = Camera('CameraDepth', PostProcessing='Depth')
    camera1.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    camera1.set_position(2.0, 0.0, 1.4)
    camera1.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera1)

    camera2 = Camera('CameraSemSeg', PostProcessing='SemanticSegmentation')
    camera2.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    camera2.set_position(2.0, 0.0, 1.4)
    camera2.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera2)

    ''' TEST camera '''

    # Base level settings
    settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=get_non_player_agents_info,
        NumberOfVehicles=NUM_VEHICLES,
        NumberOfPedestrians=NUM_PEDESTRIANS,
        SeedVehicles=SEED_VEHICLES,
        SeedPedestrians=SEED_VEHICLES,
        WeatherId=SIMWEATHER,
        QualityLevel=args.quality_level)
    return settings



class Timer(object):
    """ Timer Class

    The steps are used to calculate FPS, while the lap or seconds since lap is
    used to compute elapsed time.
    """
    def __init__(self, period):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()
        self._period_for_lap = period

    def tick(self):
        self.step += 1

    def has_exceeded_lap_period(self):
        if self.elapsed_seconds_since_lap() >= self._period_for_lap:
            return True
        else:
            return False

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) /\
                     self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time

def get_current_pose(measurement):
    """Obtains current x,y,yaw pose from the client measurements

    Obtains the current x,y, and yaw pose from the client measurements.

    Args:
        measurement: The CARLA client measurements (from read_data())

    Returns: (x, y, yaw)
        x: X position in meters
        y: Y position in meters
        yaw: Yaw position in radians
    """
    x   = measurement.player_measurements.transform.location.x
    y   = measurement.player_measurements.transform.location.y
    yaw = math.radians(measurement.player_measurements.transform.rotation.yaw)

    return (x, y, yaw)
'''TEST CAMERA'''
def print_measurements(measurements):
    #number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = 'Vehicle at ({pos_x:.1f}, {pos_y:.1f})'
   # message += '{speed:.0f} km/h, '
   # message += 'Collision: {{vehicles={col_cars:.0f}, pedestrians={col_ped:.0f}, other={col_other:.0f}}}, '
   # message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road, '
   # message += '({agents_num:d} non-player agents in the scene)'

    message = message.format(
        pos_x=player_measurements.transform.location.x,
        pos_y=player_measurements.transform.location.y,
        #speed=player_measurements.forward_speed * 3.6, # m/s -> km/h
        #col_cars=player_measurements.collision_vehicles,
        #col_ped=player_measurements.collision_pedestrians,
        #col_other=player_measurements.collision_other,
        #other_lane=100 * player_measurements.intersection_otherlane,
        #offroad=100 * player_measurements.intersection_offroad,
        #agents_num=number_of_agents)
        )
    print_over_same_line(message)
'''TEST CAMERA'''

def get_start_pos(scene):
    """Obtains player start x,y, yaw pose from the scene

    Obtains the player x,y, and yaw pose from the scene.

    Args:
        scene: The CARLA scene object

    Returns: (x, y, yaw)
        x: X position in meters
        y: Y position in meters
        yaw: Yaw position in radians
    """
    x = scene.player_start_spots[0].location.x
    y = scene.player_start_spots[0].location.y
    yaw = math.radians(scene.player_start_spots[0].rotation.yaw)

    return (x, y, yaw)

def get_player_collided_flag(measurement,
                             prev_collision_vehicles,
                             prev_collision_pedestrians,
                             prev_collision_other):
    """Obtains collision flag from player. Check if any of the three collision
    metrics (vehicles, pedestrians, others) from the player are true, if so the
    player has collided to something.

    Note: From the CARLA documentation:

    "Collisions are not annotated if the vehicle is not moving (<1km/h) to avoid
    annotating undesired collision due to mistakes in the AI of non-player
    agents."
    """
    player_meas = measurement.player_measurements
    current_collision_vehicles = player_meas.collision_vehicles
    current_collision_pedestrians = player_meas.collision_pedestrians
    current_collision_other = player_meas.collision_other

    collided_vehicles = current_collision_vehicles > prev_collision_vehicles
    collided_pedestrians = current_collision_pedestrians > \
                           prev_collision_pedestrians
    collided_other = current_collision_other > prev_collision_other

    return (collided_vehicles or collided_pedestrians or collided_other,
            current_collision_vehicles,
            current_collision_pedestrians,
            current_collision_other)

def send_control_command(client, throttle, steer, brake,
                         hand_brake=False, reverse=False):
    """Send control command to CARLA client.

    Send control command to CARLA client.

    Args:
        client: The CARLA client object
        throttle: Throttle command for the sim car [0, 1]
        steer: Steer command for the sim car [-1, 1]
        brake: Brake command for the sim car [0, 1]
        hand_brake: Whether the hand brake is engaged
        reverse: Whether the sim car is in the reverse gear
    """
    control = VehicleControl()
    # Clamp all values within their limits
    steer = np.fmax(np.fmin(steer, 1.0), -1.0)
    throttle = np.fmax(np.fmin(throttle, 1.0), 0)
    brake = np.fmax(np.fmin(brake, 1.0), 0)

    control.steer = steer
    control.throttle = throttle
    control.brake = brake
    control.hand_brake = hand_brake
    control.reverse = reverse
    client.send_control(control)

def create_controller_output_dir(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

def store_trajectory_plot(graph, fname):
    """ Store the resulting plot.
    """
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)

    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, fname)
    graph.savefig(file_name)

def write_trajectory_file(x_list, y_list, v_list, t_list, collided_list):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'trajectory.txt')

    with open(file_name, 'w') as trajectory_file:
        for i in range(len(x_list)):
            trajectory_file.write('%3.3f, %3.3f, %2.3f, %6.3f %r\n' %\
                                  (x_list[i], y_list[i], v_list[i], t_list[i],
                                   collided_list[i]))

def write_collisioncount_file(collided_list):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'collision_count.txt')

    with open(file_name, 'w') as collision_file:
        collision_file.write(str(sum(collided_list)))

def get_parkedcar_box_pts(file_path):
    # Parked car(s) (X(m), Y(m), Z(m), Yaw(deg), RADX(m), RADY(m), RADZ(m))
    parkedcar_data = None
    parkedcar_box_pts = []  # List to store the box points of the parked cars

    # Read parked car data from file
    with open(file_path, 'r') as parkedcar_file:
        next(parkedcar_file)  # Skip the header
        parkedcar_reader = csv.reader(parkedcar_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        parkedcar_data = list(parkedcar_reader)

        # Convert yaw angles to radians
        for i in range(len(parkedcar_data)):
            parkedcar_data[i][3] = parkedcar_data[i][3] * np.pi / 180.0

    # Obtain parked car(s) box points for LP
    for i in range(len(parkedcar_data)):
        x = parkedcar_data[i][0]
        y = parkedcar_data[i][1]
        z = parkedcar_data[i][2]
        yaw = parkedcar_data[i][3]
        xrad = parkedcar_data[i][4]
        yrad = parkedcar_data[i][5]
        zrad = parkedcar_data[i][6]

        # Define corner positions of the car's box in local coordinates
        cpos = np.array([
                [-xrad, -xrad, -xrad, 0,    xrad, xrad, xrad,  0],
                [-yrad, 0,     yrad,  yrad, yrad, 0,    -yrad, -yrad]])

        # Rotation matrix for yaw
        rotyaw = np.array([
                [np.cos(yaw), np.sin(yaw)],
                [-np.sin(yaw), np.cos(yaw)]])

        # Shift box points to the car's position
        cpos_shift = np.array([
                [x, x, x, x, x, x, x, x],
                [y, y, y, y, y, y, y, y]])

        # Rotate and shift the box points
        cpos = np.add(np.matmul(rotyaw, cpos), cpos_shift)

        # Append each corner point to the parkedcar_box_pts list
        for j in range(cpos.shape[1]):
            parkedcar_box_pts.append([cpos[0, j], cpos[1, j]])

    return parkedcar_box_pts

def get_stop_sign(file_path):
    #############################################
    # Load stop sign and parked vehicle parameters
    # Convert to input params for LP
    #############################################
    # Stop sign (X(m), Y(m), Z(m), Yaw(deg))
    stopsign_data = None
    stopsign_fences = []     # [x0, y0, x1, y1]
    with open(file_path, 'r') as stopsign_file:
        next(stopsign_file)  # skip header
        stopsign_reader = csv.reader(stopsign_file,
                                     delimiter=',',
                                     quoting=csv.QUOTE_NONNUMERIC)
        stopsign_data = list(stopsign_reader)
        # convert to rad
        for i in range(len(stopsign_data)):
            stopsign_data[i][3] = stopsign_data[i][3] * np.pi / 180.0
    # obtain stop sign fence points for LP
    for i in range(len(stopsign_data)):
        x = stopsign_data[i][0]
        y = stopsign_data[i][1]
        z = stopsign_data[i][2]
        yaw = stopsign_data[i][3] + np.pi / 2.0  # add 90 degrees for fence
        spos = np.array([
                [0, 0                       ],
                [0, C4_STOP_SIGN_FENCELENGTH]])
        rotyaw = np.array([
                [np.cos(yaw), np.sin(yaw)],
                [-np.sin(yaw), np.cos(yaw)]])
        spos_shift = np.array([
                [x, x],
                [y, y]])
        spos = np.add(np.matmul(rotyaw, spos), spos_shift)
        stopsign_fences.append([spos[0,0], spos[1,0], spos[0,1], spos[1,1]])
    return stopsign_fences
''' this def does not  work it does not take into account the difference between the coordinates frame of the
    carla and YOLO boxes (yolo camera)
def save_detected_car_boxes(boxes, classids, output_file=C4_PARKED_CAR_FILE):
    """
    Saves detected car bounding boxes to a text file.
    Only writes boxes corresponding to cars (classid == 2).

    Parameters:
    - frame_obj_to_detect: The frame where detection happens.
    - boxes: Detected bounding boxes.
    - classids: Detected class IDs.
    - output_file: File to store the detected car boxes (default 'detected_car_boxes.txt').
    """
    # Open the file to write the detected car bounding boxes
    with open(output_file, 'w') as file:
        # Write header
        file.write("X(m), Y(m), Z(m), YAW(deg), BOX_X_RADIUS(m), BOX_Y_RADIUS(m), BOX_Z_RADIUS(m)\n")

        # Loop through detected objects and save only cars
        for i, classid in enumerate(classids):
            if classid == 2:  # Car detected
                x, y, w, h = boxes[i]

                # Convert the bounding box data to match parked car format
                # Assuming Z 38.10 and Yaw 180.0, and radius in X and Y are half of width and height
                file.write(f"{x}, {y}, 38.10, 180.0, {w/2}, {h/2}, 0\n")
'''


def generate_accessible_colors(num_colors):
    colors = []
    for i in range(num_colors):
        # Evenly space hues for distinguishable colors
        hue = i / num_colors
        saturation = 0.7  # Keep saturation high enough for vivid colors
        lightness = 0.5   # Medium lightness for good contrast (neither too dark nor too light)

        # Convert HSL to RGB
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)

        # Convert RGB from 0-1 range to 0-255 range and cast to uint8
        rgb = tuple(int(255 * x) for x in rgb)
        colors.append(rgb)

    return np.array(colors, dtype='uint8')


def exec_waypoint_nav_demo(args):
    """ Executes waypoint navigation demo.
    """

    with make_carla_client(args.host, args.port) as client:
        # Directory to save output files
        output_dir = "output_frames"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Initialize the performance metrics tracker
        metrics = PerformanceMetrics("metrics_output")

        print('Carla client connected.')

        # Initialize the detection client
        yolo_detector = DetectionClient(host="localhost", port=5555)
        labels = yolo_detector.labels

        # Initialize the threaded detector wrapper
        threaded_detector = ThreadedDetector(yolo_detector)

        # weather
        metrics.update_weather_condition(SIMWEATHER)  # SIMWEATHER should be 6 for HARDRAINNOON

        # Initialize frame counter for FPS display
        frame_count = 0

        settings = make_carla_settings(args)

        # Now we load these settings into the server. The server replies
        # with a scene description containing the available start spots for
        # the player. Here we can provide a CarlaSettings object or a
        # CarlaSettings.ini file as string.
        scene = client.load_settings(settings)

        # Refer to the player start folder in the WorldOutliner to see the
        # player start information
        player_start = PLAYER_START_INDEX

        # client.start_episode(player_start)

        # time.sleep(CLIENT_WAIT_TIME)

        # Notify the server that we want to start the episode at the
        # player_start index. This function blocks until the server is ready
        # to start the episode.
        print('Starting new episode at %r...' % scene.map_name)
        client.start_episode(player_start)

        '''CV2 TEST'''
        '''
        measurement_data, sensor_data = client.read_data()

        print_measurements(measurement_data)
         # Save the images to disk if requested.
         #if 1==1:
        image = sensor_data['CameraRGB']

        camera_data = {'image': np.zeros((image.height, image.width, 4))}

            #img = Image.open(image)
            #img.show()
            # https://github.com/carla-simulator/carla/issues/144
            #print("image: hight:", image.height)
        #image.listen(lambda image: camera_callback(image, camera_data))

        cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RGB Camera', image)
        '''
        '''
        cv2.waitKey(1)
        while True:
             cv2.imshow('RGB Camera', camera_data['image'])

             if cv2.waitKey(1) == ord('q'):
                 break

        cv2.destroyAllWindows ()
        '''


        '''CV2 TEST'''
        #############################################
        # Load Configurations
        #############################################

        # Load configuration file (options.cfg) and then parses for the various
        # options. Here we have two main options:
        # live_plotting and live_plotting_period, which controls whether
        # live plotting is enabled or how often the live plotter updates
        # during the simulation run.
        config = configparser.ConfigParser()
        config.read(os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'options.cfg'))
        demo_opt = config['Demo Parameters']

        # Get options
        enable_live_plot = demo_opt.get('live_plotting', 'true').capitalize()
        enable_live_plot = enable_live_plot == 'True'
        live_plot_period = float(demo_opt.get('live_plotting_period', 0))

        # Set options
        live_plot_timer = Timer(live_plot_period)

        stopsign_fences = get_stop_sign(C4_STOP_SIGN_FILE)
        #stopsign_fences = None
        '''
        #############################################
        # Load stop sign and parked vehicle parameters
        # Convert to input params for LP
        #############################################
        # Stop sign (X(m), Y(m), Z(m), Yaw(deg))
        stopsign_data = None
        stopsign_fences = []     # [x0, y0, x1, y1]
        with open(C4_STOP_SIGN_FILE, 'r') as stopsign_file:
            next(stopsign_file)  # skip header
            stopsign_reader = csv.reader(stopsign_file,
                                         delimiter=',',
                                         quoting=csv.QUOTE_NONNUMERIC)
            stopsign_data = list(stopsign_reader)
            # convert to rad
            for i in range(len(stopsign_data)):
                stopsign_data[i][3] = stopsign_data[i][3] * np.pi / 180.0

        # obtain stop sign fence points for LP
        for i in range(len(stopsign_data)):
            x = stopsign_data[i][0]
            y = stopsign_data[i][1]
            z = stopsign_data[i][2]
            yaw = stopsign_data[i][3] + np.pi / 2.0  # add 90 degrees for fence
            spos = np.array([
                    [0, 0                       ],
                    [0, C4_STOP_SIGN_FENCELENGTH]])
            rotyaw = np.array([
                    [np.cos(yaw), np.sin(yaw)],
                    [-np.sin(yaw), np.cos(yaw)]])
            spos_shift = np.array([
                    [x, x],
                    [y, y]])
            spos = np.add(np.matmul(rotyaw, spos), spos_shift)
            stopsign_fences.append([spos[0,0], spos[1,0], spos[0,1], spos[1,1]])
        '''
        '''
        # Parked car(s) (X(m), Y(m), Z(m), Yaw(deg), RADX(m), RADY(m), RADZ(m))
        parkedcar_data = None
        parkedcar_box_pts = []      # [x,y]
        with open(C4_PARKED_CAR_FILE, 'r') as parkedcar_file:
            next(parkedcar_file)  # skip header
            parkedcar_reader = csv.reader(parkedcar_file,
                                          delimiter=',',
                                          quoting=csv.QUOTE_NONNUMERIC)
            parkedcar_data = list(parkedcar_reader)
            # convert to rad
            for i in range(len(parkedcar_data)):
                parkedcar_data[i][3] = parkedcar_data[i][3] * np.pi / 180.0

        # obtain parked car(s) box points for LP
        for i in range(len(parkedcar_data)):
            x = parkedcar_data[i][0]
            y = parkedcar_data[i][1]
            z = parkedcar_data[i][2]
            yaw = parkedcar_data[i][3]
            xrad = parkedcar_data[i][4]
            yrad = parkedcar_data[i][5]
            zrad = parkedcar_data[i][6]
            cpos = np.array([
                    [-xrad, -xrad, -xrad, 0,    xrad, xrad, xrad,  0    ],
                    [-yrad, 0,     yrad,  yrad, yrad, 0,    -yrad, -yrad]])
            rotyaw = np.array([
                    [np.cos(yaw), np.sin(yaw)],
                    [-np.sin(yaw), np.cos(yaw)]])
            cpos_shift = np.array([
                    [x, x, x, x, x, x, x, x],
                    [y, y, y, y, y, y, y, y]])
            cpos = np.add(np.matmul(rotyaw, cpos), cpos_shift)
            for j in range(cpos.shape[1]):
                parkedcar_box_pts.append([cpos[0,j], cpos[1,j]])

        '''
        parkedcar_box_pts = get_parkedcar_box_pts(C4_PARKED_CAR_FILE)
        #############################################
        # Load Waypoints:
        #############################################
        # Opens the waypoint file and stores it to "waypoints"
        waypoints_file = WAYPOINTS_FILENAME
        waypoints_filepath =\
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                             WAYPOINTS_FILENAME)
        waypoints_np   = None
        with open(waypoints_filepath) as waypoints_file_handle:
            waypoints = list(csv.reader(waypoints_file_handle,
                                        delimiter=',',
                                        quoting=csv.QUOTE_NONNUMERIC))
            waypoints_np = np.array(waypoints)

        #############################################
        # Controller 2D Class Declaration
        #############################################
        # This is where we take the controller2d.py class
        # and apply it to the simulator
        controller = controller2d.Controller2D(waypoints)

        #############################################
        # Determine simulation average timestep (and total frames)
        #############################################
        # Ensure at least one frame is used to compute average timestep
        num_iterations = ITER_FOR_SIM_TIMESTEP
        if (ITER_FOR_SIM_TIMESTEP < 1):
            num_iterations = 1
        '''TEST CAMERA
        SIMULATION_TIME_STEP = sim_duration / float(num_iterations)
        print("SERVER SIMULATION STEP APPROXIMATION: " + \
              str(SIMULATION_TIME_STEP))
        TOTAL_EPISODE_FRAMES = int((TOTAL_RUN_TIME + WAIT_TIME_BEFORE_START) /\
                               SIMULATION_TIME_STEP) + TOTAL_FRAME_BUFFER
        TEST CAMERA'''
        # Gather current data from the CARLA server. This is used to get the
        # simulator starting game time. Note that we also need to
        # send a command back to the CARLA server because synchronous mode
        # is enabled.

        measurement_data, sensor_data = client.read_data()


        sim_start_stamp = measurement_data.game_timestamp / 1000.0
        # Send a control command to proceed to next iteration.
        # This mainly applies for simulations that are in synchronous mode.
        send_control_command(client, throttle=0.0, steer=0, brake=1.0)
        # Computes the average timestep based on several initial iterations
        sim_duration = 0
        for i in range(num_iterations):
            # Gather current data
            measurement_data, sensor_data = client.read_data()
            # Send a control command to proceed to next iteration
            send_control_command(client, throttle=0.0, steer=0, brake=1.0)
            # Last stamp
            if i == num_iterations - 1:
                sim_duration = measurement_data.game_timestamp / 1000.0 -\
                               sim_start_stamp

        # Outputs average simulation timestep and computes how many frames
        # will elapse before the simulation should end based on various
        # parameters that we set in the beginning.
        SIMULATION_TIME_STEP = sim_duration / float(num_iterations)
        #print("SERVER SIMULATION STEP APPROXIMATION: " + \
        #      str(SIMULATION_TIME_STEP))
        TOTAL_EPISODE_FRAMES = int((TOTAL_RUN_TIME + WAIT_TIME_BEFORE_START) /\
                               SIMULATION_TIME_STEP) + TOTAL_FRAME_BUFFER
        '''TEST CAMERA
        ##if True:
        ##    for name, measurement in sensor_data.items():
        ##        filename = args.out_filename_format.format(TOTAL_EPISODE_FRAMES, name, num_iterations)
        ##        measurement.save_to_disk(filename)

        for frame in range(0, TOTAL_EPISODE_FRAMES):
         # Read the data produced by the server this frame.
            measurements, sensor_data = client.read_data()
         # Print some of the measurements.
         #if 1==1:
         ######if args.save_images_to_disk:
            if True:
                for name, measurement in sensor_data.items():
                    filename = args.out_filename_format.format(TOTAL_EPISODE_FRAMES, name, frame)
                    measurement.save_to_disk(filename)
         # We can access the encoded data of a given image as numpy
         # array using its "data" property. For instance, to get the
         # depth value (normalized) at pixel X, Y
         #
         #     depth_array = sensor_data['CameraDepth'].data
         #     value_at_pixel = depth_array[Y, X]
         #
         # Now we have to send the instructions to control the vehicle.
         # If we are in synchronous mode the server will pause the
         # simulation until we send this control.
        TEST CAMERA  '''
        #############################################
        # Frame-by-Frame Iteration and Initialization
        #############################################
        # Store pose history starting from the start position
        measurement_data, sensor_data = client.read_data()
        start_timestamp = measurement_data.game_timestamp / 1000.0
        start_x, start_y, start_yaw = get_current_pose(measurement_data)
        send_control_command(client, throttle=0.0, steer=0, brake=1.0)
        x_history     = [start_x]
        y_history     = [start_y]
        yaw_history   = [start_yaw]
        time_history  = [0]
        speed_history = [0]
        collided_flag_history = [False]  # assume player starts off non-collided

        #############################################
        # Vehicle Trajectory Live Plotting Setup
        #############################################
        # Uses the live plotter to generate live feedback during the simulation
        # The two feedback includes the trajectory feedback and
        # the controller feedback (which includes the speed tracking).
        lp_traj = lv.LivePlotter(tk_title="Trajectory Trace")
        lp_1d = lv.LivePlotter(tk_title="Controls Feedback")

        ###
        # Add 2D position / trajectory plot
        ###
        trajectory_fig = lp_traj.plot_new_dynamic_2d_figure(
                title='Vehicle Trajectory',
                figsize=(FIGSIZE_X_INCHES, FIGSIZE_Y_INCHES),
                edgecolor="black",
                rect=[PLOT_LEFT, PLOT_BOT, PLOT_WIDTH, PLOT_HEIGHT])

        trajectory_fig.set_invert_x_axis() # Because UE4 uses left-handed
                                           # coordinate system the X
                                           # axis in the graph is flipped
        trajectory_fig.set_axis_equal()    # X-Y spacing should be equal in size

        # Add waypoint markers
        trajectory_fig.add_graph("waypoints", window_size=waypoints_np.shape[0],
                                 x0=waypoints_np[:,0], y0=waypoints_np[:,1],
                                 linestyle="-", marker="", color='g')
        # Add trajectory markers
        trajectory_fig.add_graph("trajectory", window_size=TOTAL_EPISODE_FRAMES,
                                 x0=[start_x]*TOTAL_EPISODE_FRAMES,
                                 y0=[start_y]*TOTAL_EPISODE_FRAMES,
                                 color=[1, 0.5, 0])
        # Add starting position marker
        trajectory_fig.add_graph("start_pos", window_size=1,
                                 x0=[start_x], y0=[start_y],
                                 marker=11, color=[1, 0.5, 0],
                                 markertext="Start", marker_text_offset=1)
        # Add end position marker
        trajectory_fig.add_graph("end_pos", window_size=1,
                                 x0=[waypoints_np[-1, 0]],
                                 y0=[waypoints_np[-1, 1]],
                                 marker="D", color='r',
                                 markertext="End", marker_text_offset=1)
        # Add car marker
        trajectory_fig.add_graph("car", window_size=1,
                                 marker="s", color='b', markertext="Car",
                                 marker_text_offset=1)
        # Add lead car information
        trajectory_fig.add_graph("leadcar", window_size=1,
                                 marker="s", color='g', markertext="Lead Car",
                                 marker_text_offset=1)
        # Add stop sign position
        trajectory_fig.add_graph("stopsign", window_size=1,
                                # x0=[stopsign_fences[0][0]], y0=[stopsign_fences[0][1]],
                                 marker="H", color="r",
                                 markertext="Stop Sign", marker_text_offset=1)
        '''
        # Add stop sign "stop line"
        trajectory_fig.add_graph("stopsign_fence", window_size=1,

                                 #x0=[stopsign_fences[0][0], stopsign_fences[0][2]],
                                 #y0=[stopsign_fences[0][1], stopsign_fences[0][3]],

                                 color="r")
        '''

        # Load parked car points
        parkedcar_box_pts_np = np.array(parkedcar_box_pts)
        trajectory_fig.add_graph("parkedcar_pts", window_size=parkedcar_box_pts_np.shape[0],
                                 #x0=parkedcar_box_pts_np[:,0], y0=parkedcar_box_pts_np[:,1],
                                 linestyle="", marker="+", color='b')

        # Add lookahead path
        trajectory_fig.add_graph("selected_path",
                                 window_size=INTERP_MAX_POINTS_PLOT,
                                 x0=[start_x]*INTERP_MAX_POINTS_PLOT,
                                 y0=[start_y]*INTERP_MAX_POINTS_PLOT,
                                 color=[1, 0.5, 0.0],
                                 linewidth=3)

        # Add local path proposals
        for i in range(NUM_PATHS):
            trajectory_fig.add_graph("local_path " + str(i), window_size=200,
                                     x0=None, y0=None, color=[0.0, 0.0, 1.0])

        ###
        # Add 1D speed profile updater
        ###
        forward_speed_fig =\
                lp_1d.plot_new_dynamic_figure(title="Forward Speed (m/s)")
        forward_speed_fig.add_graph("forward_speed",
                                    label="forward_speed",
                                    window_size=TOTAL_EPISODE_FRAMES)
        forward_speed_fig.add_graph("reference_signal",
                                    label="reference_Signal",
                                    window_size=TOTAL_EPISODE_FRAMES)

        # Add throttle signals graph
        throttle_fig = lp_1d.plot_new_dynamic_figure(title="Throttle")
        throttle_fig.add_graph("throttle",
                              label="throttle",
                              window_size=TOTAL_EPISODE_FRAMES)
        # Add brake signals graph
        brake_fig = lp_1d.plot_new_dynamic_figure(title="Brake")
        brake_fig.add_graph("brake",
                              label="brake",
                              window_size=TOTAL_EPISODE_FRAMES)
        # Add steering signals graph
        steer_fig = lp_1d.plot_new_dynamic_figure(title="Steer")
        steer_fig.add_graph("steer",
                              label="steer",
                              window_size=TOTAL_EPISODE_FRAMES)

        # live plotter is disabled, hide windows
        if not enable_live_plot:
            lp_traj._root.withdraw()
            lp_1d._root.withdraw()


        #############################################
        # Local Planner Variables
        #############################################
        wp_goal_index   = 0
        local_waypoints = None
        path_validity   = np.zeros((NUM_PATHS, 1), dtype=bool)
        lp = local_planner.LocalPlanner(NUM_PATHS,
                                        PATH_OFFSET,
                                        CIRCLE_OFFSETS,
                                        CIRCLE_RADII,
                                        PATH_SELECT_WEIGHT,
                                        TIME_GAP,
                                        A_MAX,
                                        SLOW_SPEED,
                                        STOP_LINE_BUFFER)
        bp = behavioural_planner.BehaviouralPlanner(BP_LOOKAHEAD_BASE, stopsign_fences, LEAD_VEHICLE_LOOKAHEAD)
        #bp = behavioural_planner.BehaviouralPlanner(BP_LOOKAHEAD_BASE, LEAD_VEHICLE_LOOKAHEAD)

        #############################################
        # Scenario Execution Loop
        #############################################
        colors = generate_accessible_colors(len(labels))


        # Iterate the frames until the end of the waypoints is reached or
        # the TOTAL_EPISODE_FRAMES is reached. The controller simulation then
        # ouptuts the results to the controller output directory.
        reached_the_end = False
        skip_first_frame = True

        # Initialize the current timestamp.
        current_timestamp = start_timestamp

        # Initialize collision history
        prev_collision_vehicles    = 0
        prev_collision_pedestrians = 0
        prev_collision_other       = 0
        count_obg_detection = 0

        for frame in range(TOTAL_EPISODE_FRAMES):
            # Gather current data from the CARLA server
            measurement_data, sensor_data = client.read_data()

            '''TEST CAMERA'''
            #print_measurements(measurement_data)
         # Save the images to disk if requested.
         #if 1==1:


            #on_car_camera = sensor_data['CAMERA']
            #camera_data = {'image': np.zeros((camera.height, camera.width, 4))}

            #img = Image.open(image)
            #img.show()
            # https://github.com/carla-simulator/carla/issues/144
            #print("image: hight:", image.height)
            #sensor_data['CameraRGB'].listen(lambda image: camera_callback(camera, camera_data))
            '''
            cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RGB Camera', camera_data['image'])
            cv2.waitKey(1)

            while True:
                cv2.imshow('RGB Camera', camera_data['image'])

                if cv2.waitKey(1) == ord('q'):
                    break

            cv2.destroyAllWindows ()
            '''


            #print("image: raw data:", image.raw_data)
            #time.sleep(2.5)
            #print("image: numpy array:", image.data)
            #time.sleep(2.5)

            #print("image: RGB array:", image)
            #time.sleep(2.5)

         ######if args.save_images_to_disk:
            #print("sensor data:", sensor_data.items())
           # time.sleep(2.5)

            #image = sensor_data['CameraRGB']
                #value_at_pixel = depth_array[Y, X]
                #
            #print("image: RGB array:", image.height)
          #  print("image: RGB array:", image.raw_data)
           # print("image: RGB array:", image.width)
            #camera = sensor_data['CameraRGB']
            #print("image: RGB array:", camera)

            #camera0.listen(lambda measurement: measurement.save_to_disk('out/%06d.png' % measurement.frame))
            #cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
            #cv2.imshow('RGB Camera', image.data)
            # Convert the image to an RGB array

            # Print the RGB array
           # darknet_process.send(bgra_array+b'\n')
            '''
            for name, measurement in sensor_data.items():
                filename = args.out_filename_format.format(TOTAL_EPISODE_FRAMES, name, frame)
                #filename = args.out_filename_format.format(np.random.randint(0, TOTAL_EPISODE_FRAMES), name, frame)
                measurement.save_to_disk(filename)
            '''
            #print(on_car_camera.data)
            #print(on_car_camera.width)
            #print(on_car_camera.height)
            #img = cv2.imread(on_car_camera.data)
            #print(type(np.array(on_car_camera.data)))
            #print('OOOOOOOOOOOOOOOOO')
            #print(type(np.ndarray(on_car_camera.data)))


            '''
            try:
                #height, width = img.shape[:2]
            except:
                raise 'Image cannot be loaded! Please check the path provided!'
            finally:
            '''
            #img = np.array(on_car_camera.data)
            #vid = cv2.VideoCapture(img)
            #vid = on_car_camera.data
            #print(vid)

            #print(type(on_car_camera.raw_data))
            #print(type(on_car_camera.data))
            #frame_obj = np.array(on_car_camera.data, dtype=np.uint8)
            #frame_obj = np.array(sensor_data['CAMERA'].data).round().astype(np.uint8)
            frame_camera = sensor_data['CAMERA']
            frame_obj_to_detect = np.array(frame_camera.data)
            #print("\nframe_camera:", frame_camera)
            # TODO CAMERAS DATA
            '''
            frame_depth = sensor_data['CameraDepth'].data
            print("\nframe_depth:", frame_depth)

            frame_semseg = sensor_data['CameraSemSeg'].data
            print("\nframe_semseg:", frame_semseg)
            '''

            #frame_obj = np.array((on_car_camera.raw_data))
            #frame_obj = np.array(on_car_camera.data)
            # TODO DATA FORMATS
            '''
            print(sensor_data) # {'CAMERA': <carla.sensor.Image object at 0x000002970B91CBA8>}
            print(sensor_data['CAMERA']) # <carla.sensor.Image object at 0x000001A50B1DCDA0>
            print(sensor_data['CAMERA'].data) # [[[137 167 200]
            print(on_car_camera.data) # [[[136 166 200]
            print(on_car_camera) # <carla.sensor.Image object at 0x000001500B54F4E0>
            print(frame_obj) # [[[137 167 200]
            print(type(frame_obj)) # <class 'numpy.ndarray'>
            '''

            #height = on_car_camera.height
            #width = on_car_camera.width

            frame_count += 1

            try:
                # Process the frame with the threaded detector
                frame_obj_detected, boxes, confidences, classids, idxs = threaded_detector.process_frame(
                    frame_obj_to_detect,
                    metrics=metrics
                )
                timestamp = (datetime.now() - metrics.init_time).total_seconds()

                # Display current FPS
                detection_fps = threaded_detector.get_fps()
                if frame_count % 10 == 0:  # Only print every 10 frames to reduce console spam
                    print(f"Detection FPS: {detection_fps:.1f}")

                # Convert from RGB to BGR for OpenCV display (if needed)
                if frame_obj_detected is not None:
                    try:
                        # Only convert if needed (check if it's RGB vs BGR)
                        #if frame_obj_detected.shape[2] == 3:  # 3 channels
                        if True:  # 3 channels
                            # Check if already in BGR format
                           # if np.array_equal(frame_obj_detected[:5, :5, :], cv2.cvtColor(cv2.cvtColor(frame_obj_detected[:5, :5, :], cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2BGR)):
                            if False:
                                # Already in BGR format
                                pass
                            else:
                                # Convert from RGB to BGR
                                frame_obj_detected = cv2.cvtColor(frame_obj_detected, cv2.COLOR_RGB2BGR)

                        # Display the frame
                        cv2.imshow('OUTPUT: OBJECT DETECTION', frame_obj_detected)
                        cv2.waitKey(1)
                    except Exception as e:
                        print(f"Warning: Display error: {e}")

                # Record risk level from the detector
                if boxes is not None and len(boxes) > 0:
                    risk_level = yolo_detector._calculate_traffic_risk(boxes, classids, confidences)
                    metrics.record_risk_level(risk_level)
                    # Record the detection results
                    metrics.record_sign_detection(timestamp, classids, confidences, boxes, idxs)

            except Exception as e:
                print(f"Detection failed: {e}")
                import traceback
                traceback.print_exc()  # Print full stack trace
                # Fall back to previous frame or empty results
                frame_obj_detected = frame_obj_to_detect
                boxes, confidences, classids, idxs = [], [], [], []

            ''' BEFORE SOCKET
            # Replace your current infer_image call with:
            frame_obj_to_detect, boxes, confidences, classids, idxs = infer_image_optimized(
                yolo_detector,
                frame_obj_to_detect,
                metrics=metrics
            )  # If you're using metrics
            '''

            '''
            if count_obg_detection == 0:
                frame_obj_to_detect, boxes, confidences, classids, idxs = infer_image(net, layer_names,
                                        sensor_data['CAMERA'].height, sensor_data['CAMERA'].width,
                                        frame_obj_to_detect, colors, labels, metrics=metrics)
                count_obg_detection += 1
            else:
                frame_obj_to_detect, boxes, confidences, classids, idxs = infer_image(net, layer_names,
                                        sensor_data['CAMERA'].height, sensor_data['CAMERA'].width,
                                        frame_obj_to_detect, colors, labels, boxes, confidences,
                                        classids, idxs, infer=False, metrics=metrics)
                count_obg_detection = (count_obg_detection + 1) % 6
            '''
            # Add this line to check for stop signs and display warning
            #frame_obj_to_detect = display_object_warnings(frame_obj_to_detect, boxes, confidences,
             #                                        classids, idxs, metrics=metrics)

            # Convert from RGB to BGR for OpenCV display
            # This section is now handled by the threaded detector code above
            #print("\nidexs:",idxs)
            #print("\nconfidences:",confidences)
            #print("\nclassids:",classids)
            #print("\nboxes:", boxes)

            # cheking if the stop sign is detected
            """ there is no def set stop sign
            if frame == 100:
                stopsign_fences = get_stop_sign(C4_STOP_SIGN_FILE)
                bp = behavioural_planner.BehaviouralPlanner(BP_LOOKAHEAD_BASE, stopsign_fences, LEAD_VEHICLE_LOOKAHEAD)
                print("STOP SIGN RELOADED", stopsign_fences)

            # TODO save parked car boxes
            """
            '''
            if classids and 2 in classids:
                # Save detected car's bounding box to a file
                save_detected_car_boxes(boxes, classids)
                parkedcar_box_pts = get_parkedcar_box_pts(C4_PARKED_CAR_FILE)
                parkedcar_box_pts_np = np.array(parkedcar_box_pts)
            '''

            '''


                with open(C4_PARKED_CAR_FILE, 'w') as file:
                    file.write("X(m), Y(m), Z(m), YAW(deg), BOX_X_RADIUS(m), BOX_Y_RADIUS(m), BOX_Z_RADIUS(m)\n")
                    for i, classid in enumerate(classids):
                        if classid == 2:  # Car detected
                            x, y, w, h = boxes[i]
                            # Example transformation of bounding box to match the parked car format
                            file.write(f"{x}, {y}, 0, 0, {w/2}, {h/2}, 0\n")  # Assuming Z and Yaw are 0 for simplicity
            '''

            '''
            # # TODO Save the images obj detected to disk.
            # https://stackoverflow.com/questions/71413891/convert-rgb-values-to-jpg-image-in-python
            filename = "./_out/output_image"
            #os.mkdir(filename)
            image_RGB = frame_obj_to_detect
            image = Image.fromarray(image_RGB.astype('uint8')).convert('RGB')
            #cv2.imwrite(filename, frame_obj) # Save the image
            image.save(f"{filename}/output_image_detected{frame}.jpg")

            # TODO Save the images to disk.

            '''
            '''
            filename = args.out_filename_format.format(TOTAL_EPISODE_FRAMES, 'frame_camera', frame)
            frame_camera.save_to_disk(filename)

            output_path = "_out/on_car_camera_depth"
            output_file = os.path.join(output_path, f"frame_{frame}.dat")
            depth_dir = os.path.dirname(output_file)
            if not os.path.exists(depth_dir):
             os.makedirs(depth_dir)
            with open(output_file, 'wb') as file:
                np.savetxt(file, frame_depth, delimiter=',', fmt='%0.8f')

            output_path = "_out/on_car_camera_semseg"
            output_file = os.path.join(output_path, f"frame_{frame}.dat")
            depth_dir = os.path.dirname(output_file)
            if not os.path.exists(depth_dir):
             os.makedirs(depth_dir)
            with open(output_file, 'wb') as file:
                np.savetxt(file, frame_semseg, delimiter=',', fmt='%d')
            '''

            '''
            filename_depth = args.out_filename_format.format(TOTAL_EPISODE_FRAMES, 'on_car_camera_depth', frame)

            depth_dir = os.path.dirname(filename_depth)
            if not os.path.exists(depth_dir):
             os.makedirs(depth_dir)
            with open(filename_depth + ".txt", "wb") as f:
                np.save(f, frame_depth)
            #frame_depth.save_to_disk(filename)
            filename_semseg = args.out_filename_format.format(TOTAL_EPISODE_FRAMES, 'on_car_camera_semseg', frame)
            semseg_dir = os.path.dirname(filename_semseg)
            if not os.path.exists(semseg_dir):
             os.makedirs(semseg_dir)
            with open(filename_semseg + ".txt", "wb") as f:
                np.save(f, frame_semseg)
            #frame_semseg.save_to_disk(filename)

            # TODO Save the infer to disk.

            # Save the data as a text file
            output_file = os.path.join(output_dir, f"frame_{frame}.txt")
            with open(output_file, "w") as f:
                f.write(f"Frame {frame}:\n")
                f.write("Boxes:\n")
                for box in boxes:
                    f.write(f"{box}\n")
                f.write("Confidences:\n")
                for confidence in confidences:
                    f.write(f"{confidence}\n")
                f.write("Class IDs:\n")
                for classid in classids:
                    f.write(f"{classid}\n")
                f.write("Indexes:\n")
                for idx in idxs:
                    f.write(f"{idx}\n")
            '''
            '''
            for name, measurement in sensor_data.items():
                print("measurement")
                print(measurement) # <carla.sensor.Image object at 0x000001910EEF1710>

                print("measurement.daata:")
                print(measurement.data) # [[[136 166 200]...

            '''


            #vid.release()

            ''' WORKING FOR IMAGES
            img = np.array(on_car_camera.data)
            img, _, _, _, _ = infer_image(net, layer_names, on_car_camera.height, on_car_camera.width, img, colors, labels)
            show_image(img)
            '''

            ######## OLD WORKING#####bgra_array = to_bgra_array(on_car_camera)
            ######## OLD WORKING#####cv2.imshow('RGB Camera', bgra_array)
            '''TEST CAMERA'''


            # Update pose and timestamp
            prev_timestamp = current_timestamp
            current_x, current_y, current_yaw = \
                get_current_pose(measurement_data)
            current_speed = measurement_data.player_measurements.forward_speed
            current_timestamp = float(measurement_data.game_timestamp) / 1000.0

            # Wait for some initial time before starting the demo
            if current_timestamp <= WAIT_TIME_BEFORE_START:
                send_control_command(client, throttle=0.0, steer=0, brake=1.0)
                continue
            else:
                current_timestamp = current_timestamp - WAIT_TIME_BEFORE_START

            # Store history
            x_history.append(current_x)
            y_history.append(current_y)
            yaw_history.append(current_yaw)
            speed_history.append(current_speed)
            time_history.append(current_timestamp)

            # Store collision history
            collided_flag,\
            prev_collision_vehicles,\
            prev_collision_pedestrians,\
            prev_collision_other = get_player_collided_flag(measurement_data,
                                                 prev_collision_vehicles,
                                                 prev_collision_pedestrians,
                                                 prev_collision_other)
            collided_flag_history.append(collided_flag)

            ###
            # Local Planner Update:
            #   This will use the behavioural_planner.py and local_planner.py
            #   implementations that the learner will be tasked with in
            #   the Course 4 final project
            ###

            # Obtain Lead Vehicle information.
            lead_car_pos    = []
            lead_car_length = []
            lead_car_speed  = []
            for agent in measurement_data.non_player_agents:
                agent_id = agent.id
                if agent.HasField('vehicle'):
                    lead_car_pos.append(
                            [agent.vehicle.transform.location.x,
                             agent.vehicle.transform.location.y])
                    lead_car_length.append(agent.vehicle.bounding_box.extent.x)
                    lead_car_speed.append(agent.vehicle.forward_speed)

            # Execute the behaviour and local planning in the current instance
            # Note that updating the local path during every controller update
            # produces issues with the tracking performance (imagine everytime
            # the controller tried to follow the path, a new path appears). For
            # this reason, the local planner (LP) will update every X frame,
            # stored in the variable LP_FREQUENCY_DIVISOR, as it is analogous
            # to be operating at a frequency that is a division to the
            # simulation frequency.
            if frame % LP_FREQUENCY_DIVISOR == 0:
                # --------------------------------------------------------------
                #  # Compute open loop speed estimate.
                open_loop_speed = lp._velocity_planner.get_open_loop_speed(current_timestamp - prev_timestamp)

                #  # Calculate the goal state set in the local frame for the local planner.
                #  # Current speed should be open loop for the velocity profile generation.
                ego_state = [current_x, current_y, current_yaw, open_loop_speed]

                #  # Set lookahead based on current speed.
                bp.set_lookahead(BP_LOOKAHEAD_BASE + BP_LOOKAHEAD_TIME * open_loop_speed)

                #  # Perform a state transition in the behavioural planner.
                bp.transition_state(waypoints, ego_state, current_speed)

                #  # Check to see if we need to follow the lead vehicle.
                bp.check_for_lead_vehicle(ego_state, lead_car_pos[1])

                #  # Compute the goal state set from the behavioural planner's computed goal state.
                # Conformal Lattice Planning
                goal_state_set = lp.get_goal_state_set(bp._goal_index, bp._goal_state, waypoints, ego_state)

                #  # Calculate planned paths in the local frame.
                paths, path_validity = lp.plan_paths(goal_state_set)

                #  # Transform those paths back to the global frame.
                paths = local_planner.transform_paths(paths, ego_state)

                #  # Perform collision checking.
                collision_check_array = lp._collision_checker.collision_check(paths, [parkedcar_box_pts])

                #  # Compute the best local path.
                best_index = lp._collision_checker.select_best_path_index(paths, collision_check_array, bp._goal_state)
                #  # If no path was feasible, continue to follow the previous best path.
                if best_index == None:
                    best_path = lp._prev_best_path
                else:
                    best_path = paths[best_index]
                    lp._prev_best_path = best_path

                #  # Compute the velocity profile for the path, and compute the waypoints.
                #  # Use the lead vehicle to inform the velocity profile's dynamic obstacle handling.
                #  # In this scenario, the only dynamic obstacle is the lead vehicle at index 1.
                desired_speed = bp._goal_state[2]
                lead_car_state = [lead_car_pos[1][0], lead_car_pos[1][1], lead_car_speed[1]]
                decelerate_to_stop = bp._state == behavioural_planner.DECELERATE_TO_STOP
                local_waypoints = lp._velocity_planner.compute_velocity_profile(best_path, desired_speed, ego_state, current_speed, decelerate_to_stop, lead_car_state, bp._follow_lead_vehicle)
                # --------------------------------------------------------------

                if local_waypoints != None:
                    # Update the controller waypoint path with the best local path.
                    # This controller is similar to that developed in Course 1 of this
                    # specialization.  Linear interpolation computation on the waypoints
                    # is also used to ensure a fine resolution between points.
                    wp_distance = []   # distance array
                    local_waypoints_np = np.array(local_waypoints)
                    for i in range(1, local_waypoints_np.shape[0]):
                        wp_distance.append(
                                np.sqrt((local_waypoints_np[i, 0] - local_waypoints_np[i-1, 0])**2 +
                                        (local_waypoints_np[i, 1] - local_waypoints_np[i-1, 1])**2))
                    wp_distance.append(0)  # last distance is 0 because it is the distance
                                           # from the last waypoint to the last waypoint

                    # Linearly interpolate between waypoints and store in a list
                    wp_interp      = []    # interpolated values
                                           # (rows = waypoints, columns = [x, y, v])
                    for i in range(local_waypoints_np.shape[0] - 1):
                        # Add original waypoint to interpolated waypoints list (and append
                        # it to the hash table)
                        wp_interp.append(list(local_waypoints_np[i]))

                        # Interpolate to the next waypoint. First compute the number of
                        # points to interpolate based on the desired resolution and
                        # incrementally add interpolated points until the next waypoint
                        # is about to be reached.
                        num_pts_to_interp = int(np.floor(wp_distance[i] /\
                                                     float(INTERP_DISTANCE_RES)) - 1)
                        wp_vector = local_waypoints_np[i+1] - local_waypoints_np[i]
                        wp_uvector = wp_vector / np.linalg.norm(wp_vector[0:2])

                        for j in range(num_pts_to_interp):
                            next_wp_vector = INTERP_DISTANCE_RES * float(j+1) * wp_uvector
                            wp_interp.append(list(local_waypoints_np[i] + next_wp_vector))
                    # add last waypoint at the end
                    wp_interp.append(list(local_waypoints_np[-1]))

                    # Update the other controller values and controls
                    controller.update_waypoints(wp_interp)
                    pass

            ###
            # Controller Update
            ###
            if local_waypoints != None and local_waypoints != []:
                controller.update_values(current_x, current_y, current_yaw,
                                         current_speed,
                                         current_timestamp, frame)
                controller.update_controls()
                cmd_throttle, cmd_steer, cmd_brake = controller.get_commands()
            else:
                cmd_throttle = 0.0
                cmd_steer = 0.0
                cmd_brake = 0.0

            metrics.record_vehicle_response(timestamp, cmd_throttle, cmd_brake,
                               cmd_steer, current_speed)

            # Skip the first frame or if there exists no local paths
            if skip_first_frame and frame == 0:
                pass
            elif local_waypoints == None:
                pass
            else:
                # Update live plotter with new feedback
                # This way, you ensure that the roll function receives scalar values, one by one, for each parked car point.
                for i in range(parkedcar_box_pts_np.shape[0]):
                    trajectory_fig.roll("parkedcar_pts", parkedcar_box_pts_np[i, 0], parkedcar_box_pts_np[i, 1])
                #trajectory_fig.roll("parkedcar_pts", parkedcar_box_pts_np[:,0], parkedcar_box_pts_np[:,1])
                trajectory_fig.roll("stopsign", stopsign_fences[0][0], stopsign_fences[0][1])
               # trajectory_fig.roll("stopsign_fence", [stopsign_fences[0][0], stopsign_fences[0][2]], [stopsign_fences[0][1], stopsign_fences[0][3]])
                trajectory_fig.roll("trajectory", current_x, current_y)
                trajectory_fig.roll("car", current_x, current_y)
                if lead_car_pos:    # If there exists a lead car, plot it
                    trajectory_fig.roll("leadcar", lead_car_pos[1][0],
                                        lead_car_pos[1][1])
                forward_speed_fig.roll("forward_speed",
                                       current_timestamp,
                                       current_speed)
                forward_speed_fig.roll("reference_signal",
                                       current_timestamp,
                                       controller._desired_speed)
                throttle_fig.roll("throttle", current_timestamp, cmd_throttle)
                brake_fig.roll("brake", current_timestamp, cmd_brake)
                steer_fig.roll("steer", current_timestamp, cmd_steer)

                # Local path plotter update
                if frame % LP_FREQUENCY_DIVISOR == 0:
                    path_counter = 0
                    for i in range(NUM_PATHS):
                        # If a path was invalid in the set, there is no path to plot.
                        if path_validity[i]:
                            # Colour paths according to collision checking.
                            if not collision_check_array[path_counter]:
                                colour = 'r'
                            elif i == best_index:
                                colour = 'k'
                            else:
                                colour = 'b'
                            trajectory_fig.update("local_path " + str(i), paths[path_counter][0], paths[path_counter][1], colour)
                            path_counter += 1
                        else:
                            trajectory_fig.update("local_path " + str(i), [ego_state[0]], [ego_state[1]], 'r')
                # When plotting lookahead path, only plot a number of points
                # (INTERP_MAX_POINTS_PLOT amount of points). This is meant
                # to decrease load when live plotting
                wp_interp_np = np.array(wp_interp)
                path_indices = np.floor(np.linspace(0,
                                                    wp_interp_np.shape[0]-1,
                                                    INTERP_MAX_POINTS_PLOT))
                trajectory_fig.update("selected_path",
                        wp_interp_np[path_indices.astype(int), 0],
                        wp_interp_np[path_indices.astype(int), 1],
                        new_colour=[1, 0.5, 0.0])


                # Refresh the live plot based on the refresh rate
                # set by the options
                if enable_live_plot and \
                   live_plot_timer.has_exceeded_lap_period():
                    lp_traj.refresh()
                    lp_1d.refresh()
                    live_plot_timer.lap()

            # Output controller command to CARLA server
            send_control_command(client,
                                 throttle=cmd_throttle,
                                 steer=cmd_steer,
                                 brake=cmd_brake)

            # Find if reached the end of waypoint. If the car is within
            # DIST_THRESHOLD_TO_LAST_WAYPOINT to the last waypoint,
            # the simulation will end.
            dist_to_last_waypoint = np.linalg.norm(np.array([
                waypoints[-1][0] - current_x,
                waypoints[-1][1] - current_y]))
            if  dist_to_last_waypoint < DIST_THRESHOLD_TO_LAST_WAYPOINT:
                reached_the_end = True
            if reached_the_end:
                break

        # End of demo - Stop vehicle and Store outputs to the controller output
        # directory.
        if reached_the_end:
            print("Reached the end of path. Writing to controller_output...")
        else:
            print("Exceeded assessment time. Writing to controller_output...")
        # Stop the car
        send_control_command(client, throttle=0.0, steer=0.0, brake=1.0)
        # Store the various outputs
        store_trajectory_plot(trajectory_fig.fig, 'trajectory.png')
        store_trajectory_plot(forward_speed_fig.fig, 'forward_speed.png')
        store_trajectory_plot(throttle_fig.fig, 'throttle_output.png')
        store_trajectory_plot(brake_fig.fig, 'brake_output.png')
        store_trajectory_plot(steer_fig.fig, 'steer_output.png')
        write_trajectory_file(x_history, y_history, speed_history, time_history,
                              collided_flag_history)
        write_collisioncount_file(collided_flag_history)

    print("Generating performance metrics summary...")
    metrics.update_weather_condition(SIMWEATHER)  # SIMWEATHER should be 6 for HARDRAINNOON
    metrics.generate_summary()
    metrics.visualize_metrics()

    print("Generating results report...")
    reporter = ResultsReporter(metrics_dir="metrics_output", report_dir="results_report")
    reporter.generate_report()

    print("Testing complete - results available in 'results_report/performance_report.html'")


#  cleanup
def cleanup():
    try:
        print("Performing cleanup...")
        # Close detector connections if they exist
        if 'yolo_detector' in globals() and yolo_detector:
            yolo_detector.close()

        # Ensure metrics are saved
        if 'metrics' in globals() and metrics:
            try:
                metrics.generate_summary()
                metrics.visualize_metrics()
                print("Performance metrics saved successfully")
            except Exception as e:
                print(f"Error saving metrics: {e}")

        # Delete any temporary files
        temp_files = glob.glob('temp_*.jpg')
        for file in temp_files:
            try:
                os.remove(file)
            except:
                pass

        print("Cleanup completed")
    except Exception as e:
        print(f"Error during cleanup: {e}")

# Register cleanup
atexit.register(cleanup)

# Register cleanup function
atexit.register(cleanup)

def main():
    """Main function.

    Args:
        -v, --verbose: print debug information
        --host: IP of the host server (default: localhost)
        -p, --port: TCP port to listen to (default: 2000)
        -a, --autopilot: enable autopilot
        -q, --quality-level: graphics quality level [Low or Epic]
        -i, --images-to-disk: save images to disk
        -c, --carla-settings: Path to CarlaSettings.ini file
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Low',
        help='graphics quality level.')
    '''test camera'''
    argparser.add_argument(
        '-i', '--images-to-disk',
        action='store_true',
        dest='save_images_to_disk',
        help='save images (and Lidar data if active) to disk')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')
    args = argparser.parse_args()

    # Logging startup info
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'

    # Execute when server connection is established
    try:
        while True:
            try:
                #check_and_install_dependencies()
                exec_waypoint_nav_demo(args)
                print("Waypoint navigation demo completed successfully")
                break
            except TCPConnectionError as error:
                logging.error(error)
                logging.info("Reconnecting to server in 1 seconds...")
                time.sleep(1)
                break

            except KeyboardInterrupt:
                print("\nCancelled by user. Bye!")
                break
            except Exception as e:
                logging.error(f"Unexpected error during execution: {e}")
                logging.info("Attempting to recover in 5 seconds...")
                traceback.print_exc()
                time.sleep(5)
                break

    finally:
        print("Cleaning up resources...")
        cleanup()

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
