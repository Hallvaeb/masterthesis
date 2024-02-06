"""trt_yolo.py
This script uses TensorRT optimized YOLO engine to detect people on images
"""

import os

# Set the PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION environment variable
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import json
import sys
import subprocess

# LOAD
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import numpy as np
import requests
from PIL import Image
import time
from datetime import datetime, timezone
from os import listdir
from os.path import isfile, join
import uuid
from io import BytesIO
from health_check import get_full_health_check

from coordinate_helper import compute_coordinates_of_shrunk_image
# from upload.dropbox_helper import upload_image
from upload.upload_schedule_reader import should_upload_image_timebased, should_upload_image
from upload.google_storage import upload_blob
from image_manipulator import gamma_correction, draw_text_on_image
from utils.camera import add_camera_args, Camera
from utils.visualization import BBoxVisualization
from utils.yolo_classes import get_cls_dict
from logger import write_to_log
import failure_count
# import pycuda.autoinit  # This is needed for initializing CUDA driver
import traceback
import hashlib
import os
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from raspberry_pi_files.yolov3.utils import detect_image, Load_Yolo_model
import picamera
import process_check

# All imports MUST be above the following import and GreenGrass installation
sys.path.append('/home/pi/config')
from config import API_ENDPOINT, API_KEY, SERIAL, TOKEN, SERIAL_NR, ZONE_ANALYSIS

WINDOW_NAME = "TrtYOLODemo"
stop_threads = False
dimension = (608, 608)
current_minutes = ""


def parse_args():
    # Parse input arguments.
    desc = (
        "Capture and display live camera video, while doing "
        "real-time object detection with TensorRT optimized "
        "YOLO model"
    )
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        "-p",
        "--with_plugins",
        action="store_true",
        help="use a TensorRT engine with yolo plugins",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov4-608",
        help=(
            "[yolov3|yolov3-tiny|yolov3-spp|yolov4|yolov4-tiny]-"
            "[{dimension}], where dimension could be a single "
            "number (e.g. 288, 416, 608) or WxH (e.g. 416x256)"
        ),
    )
    parser.add_argument(
        "--category_num", type=int, default=80, help="number of object categories [80]"
    )
    parser.add_argument(
        "-t",
        "--type",
        type=int,
        default=0,
        help="0 = on a timer (default), 1 = press to capture/detect, 2 = captures/detects in succession",
    )
    parser.add_argument(
        "-c",
        "--confidence",
        type=int,
        default=70,
        help="Confidence threshhold. Filter detections with confindence below this value",
    )
    return parser.parse_args()


def push_image(img, count: int, timestamp, image_filename: str):
    """
    Uploads image to bucket. Sends count to endpoint
    @param img:image as numpy array
    @param count: count of detected persons in the image
    """
    should_upload = False

    try:
        should_upload = should_upload_image()
    except:
        global current_minutes
        should_upload = should_upload_image_timebased(current_minutes)

    if should_upload:		
        # convert BGR to RGB and right format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        buffered2 = BytesIO()
        img_pil.save(buffered2, format="JPEG")

        try:
            upload_blob("hallmonitor_3_input", image_filename, buffered2)
        except Exception as e:
            print("Failed to upload image to cloud: " + e, 1)

    DEVICE_ID = hashlib.sha256((SERIAL + TOKEN).encode('utf-8')).hexdigest()
    data = {
        "key": API_KEY,
        "file_name": image_filename,
        "count": count,
        "device_id": DEVICE_ID,
        "timestamp": timestamp
    }

    upload_analytic_results(data)
    upload_health_data()
 
def upload_detection_data(filename, detections, timestamp):
        DEVICE_ID = hashlib.sha256((SERIAL + TOKEN).encode('utf-8')).hexdigest()
        ngrok_test_ip = f"https://20220825-pa.hallmonitor.eu/detections?device_id={DEVICE_ID}&token={API_KEY}"
        headers = {"Content-Type": "application/json; charset=utf-8"}

        detections = json.dumps([detection.__dict__ for detection in detections])
        data = {"device_id": DEVICE_ID, "file": filename, "detections": detections, "created_at": timestamp}

        http = requests.Session()
        response = http.post(url=ngrok_test_ip, json=data, headers=headers)


def upload_health_data():
    data = get_full_health_check()

    retry_strategy = Retry(
        total=5,
        backoff_factor=0.1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)

    # Device Health Endpoint
    ENDPOINT = 'https://prod01.api.hallmonitor.dk/pirapi/api.php/outdoor/device_health'

    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)
    try:
        response = http.post(url=ENDPOINT, data=data)
        if response.status_code == 201:
            elapsedTime = response.elapsed
            write_to_log(str(response.status_code) + ": Device health data successfully uploaded ({0} ms)".format(
                (elapsedTime.microseconds / 1000)), True)
        else:
            write_to_log(str(response.status_code) + ": Device health data not uploaded", True)
    except Exception as e:
        write_to_log('Caught error. Stopping cam\n' + str(e.__class__) + 'occured\n', True)
        write_to_log(traceback.format_exc(), True)


def upload_analytic_results(data):
    # Retry strategy for pushing to Hallmonitor backend
    retry_strategy = Retry(
        total=5,
        backoff_factor=0.1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)

    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)
    try:
        response = http.post(url=API_ENDPOINT, data=data)
        if response.status_code == 201:
            elapsedTime = response.elapsed
            write_to_log(str(response.status_code) + ": Analytic values successfully uploaded ({0} ms)".format(
                (elapsedTime.microseconds / 1000)), True)
        else:
            write_to_log(str(response.status_code) + ": Analytic not uploaded", True)
    except Exception as e:
        write_to_log('Caught error. Stopping cam\n' + str(e.__class__) + 'occured\n', True)
        write_to_log(traceback.format_exc(), True)


def detect_persons(numpyImage, trt_yolo, conf_th):
    """
    @param numpyImage: image as numpy array
    @param trt_yolo: instance of TrtYolo class
    @param conf_th: Confidence threshold
    @return: list of detections (boxes, class, confidence)
    """
    # boxes, confs, clss = trt_yolo.detect(numpyImage, conf_th)
    boxes, confs, clss = detect_image(Yolo=trt_yolo, numpyImage=numpyImage, score_threshold=conf_th, output_path='')
    person_boxes = []
    clss_filtered = []
    conf_filtered = []
    count = 0
    for i in range(len(boxes)):
        print("Box, class: ", i, boxes[i], clss[i], confs[i])
        if clss[i] == 0:
            count += 1
            person_boxes.append(boxes[i])
            clss_filtered.append(clss[i])
            conf_filtered.append(confs[i])

    return person_boxes, clss_filtered, conf_filtered

def take_picture(img, trt_yolo, conf_th, vis):
    """
    @param img: image captured already
    @param trt_yolo: TrtYolo instance
    @param conf_th: Confidence threshold
    @param vis: Visualization instance
    """
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    timestamp_hallvard = datetime.now().strftime("%d%m%y-%H%M%S")

    img = gamma_correction(img)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    write_to_log("Image gamma corrected")

    # detect on full image
    person_boxes_full, clss_filtered_full, conf_filtered_full = detect_persons(
        img, trt_yolo, conf_th
    )

    detections_full_img = wrap_detections_values(
        person_boxes_full, conf_filtered_full, clss_filtered_full
    )

    # blur image
    img = cv2.GaussianBlur(img, (211, 211), 0)

    # Compress image
    divide_amount = 4
    img = shrink_image(img, divide_amount)
    compute_coordinates_of_shrunk_image(detections_full_img, divide_amount)

    # img = vis.draw_bboxes(img, detections_flat, 0)
    img = vis.draw_bboxes(img, detections_full_img, 1)

    """Debugging purposes for pushing detection coordinates"""
    aNumber = 0

    for detection in detections_full_img:
            # print(aNumber, "Detection (Full-image): ", detection)
            print("Confidence: ", detection.conf)
            print("Box coords: ", detection.box)
            print("Class: ", detection.cl)
            aNumber = aNumber + 1

    count = len(detections_full_img)
    image_filename = f"hallvard-{str(timestamp_hallvard)}-{str(count)}.jpg"

    # Hallvard
    push_image(img, count, timestamp, image_filename)  # push to api using post request


def wrap_detections_values(boxes, confs, clss):
    detections = []
    for i in range(len(boxes)):
        detection = Detection(boxes[i], confs[i], clss[i])
        detections.append(detection)
    return detections

# (--type 0) default. analyzes one picture and exits script
def start(img_ndarray, vis, trt_yolo, conf_th):
    take_picture(img_ndarray, trt_yolo, conf_th, vis)

# (--type 1) analyaze next picture when enter is pressed.
# images can be captured by onboard cam or fed as singles or multiple via folder through the terminal. this is only used for testing
def start_manual(cam, vis, trt_yolo, images, conf_th, horizontal_tile_amount, vertical_tile_amount):
	run = True
	i = 0
	while run:
		print("Enter to capture. Enter q to quit")
		user_input = input()
		if user_input.lower() != "q":
			if images is not None:
				if i < len(images):
					cam.set_handle(images[i])
					i += 1
				else:
					print("No more pictures")
					break
			take_picture(cam, trt_yolo, conf_th, vis, horizontal_tile_amount, vertical_tile_amount)
			print("Captured")
		else:
			break


# (--type 2) analyze multiple images in succession automatically
# images can be captured by onboard cam or fed as singles or multiple via folder through the terminal. this is only used for testing
def analyze_multiple(cam, vis, trt_yolo, images, conf_th, horizontal_tile_amount, vertical_tile_amount):
	i = 0
	progress = 0.0
	write_to_log("Analyzing image")
	while i < 100:
		if images is not None:
			if i < len(images):
				cam.set_handle(images[i])
				progress = (i + 1) / len(images) * 100
			else:
				break
		print("Analyzing image", str(i + 1))
		take_picture(cam, trt_yolo, conf_th, vis, horizontal_tile_amount, vertical_tile_amount)
		print("\n\n\n", str(progress), "%\n\n\n")
		i += 1



def initiate_camera(args):
    write_to_log("Starting camera")
    try:
        cam = Camera(args)
        cam.open()

        # if not cam.is_opened:
        # write_to_log('ERROR: failed to open camera', True)
        # raise Exception("ERROR: failed to open camera!")
        print(cam.is_opened)
        cam.start()
    except:
        write_to_log('ERROR: failed to open camera', True)
        raise Exception("ERROR: failed to open camera!")
    return cam

def operation_mode(args, cam, vis, trt_yolo, conf_th, horizontal_tile_amount, vertical_tile_amount):
	images = None

	if args.use_folder:
		filename = args.filename
		images = [
			filename + "/" + f for f in listdir(filename) if isfile(join(filename, f))
		]
	elif args.use_image:
		images = [args.filename]

	global stop_threads

	time.sleep(1)  # crashes occur if an image is captured immediately after opening
	if args.type == 0:
		start(cam, vis, trt_yolo, conf_th, horizontal_tile_amount, vertical_tile_amount)
		""" print("Waiting..")
        thread = threading.Thread(
            target=start_schedule, args=[cam, vis, trt_yolo, conf_th]
        )
        thread.start()
        input("Press enter to stop")

        stop_threads = True
        thread.join() """
	elif args.type == 1:
		start_manual(cam, vis, trt_yolo, images, conf_th, horizontal_tile_amount, vertical_tile_amount)
	elif args.type == 2:
		analyze_multiple(cam, vis, trt_yolo, images, conf_th, horizontal_tile_amount, vertical_tile_amount)

# shrinks image img (numpy array) based on divide_amount value
def shrink_image(img, divide_amount):
    # img = cv2.resize(img, dsize=(1920, 1080), interpolation=cv2.INTER_NEAREST)
    new_width = img.shape[1] / divide_amount
    new_height = img.shape[0] / divide_amount

    new_width_int = int(round(new_width))
    new_height_int = int(round(new_height))

    img = cv2.resize(
        img, dsize=(new_width_int, new_height_int), interpolation=cv2.INTER_NEAREST
    )
    return img

def capture_image_pi(image_rot):
    with picamera.PiCamera() as camera:
        camera.resolution = (3264, 2464)
        camera.framerate = 24
        camera.rotation = image_rot*90
        time.sleep(2)
        image = np.empty((2464 * 3264 * 3,), dtype=np.uint8)
        camera.capture(image, 'bgr')
        image = image.reshape((2464, 3264, 3))
    return image


def main():

    process_check.processKiller()

    fail_count = failure_count.get_value()  # if the script failed a bunch of times in a row it
    if fail_count > 1:  # should reboot. e.g reboot clears memory
        os.system('sudo reboot')

    now = datetime.now()

    global current_minutes
    current_minutes = int(now.strftime("%M"))
    args = parse_args()
    camera_rotated = args.cam_flip
    conf_th = args.confidence / 100.0

    trt_yolo = Load_Yolo_model()
    cls_dict = get_cls_dict(args.category_num)

    img_ndarray = capture_image_pi(camera_rotated)
    vis = BBoxVisualization(cls_dict)
    operation_mode(args, img_ndarray, vis, trt_yolo, conf_th)


class Detection:
    def __init__(self, box, conf, cl):
        self.box = box
        self.conf = conf
        self.cl = cl

    def __str__(self):
        return str(self.box) + "\n" + str(self.conf) + "\n" + str(self.cl)

def just_rebooted():
    #This func returns true if detect should run.
    #It returns false if the device has just been rebooted at the scheduled time.
    path = '/home/ggc_user/reboot_at_night_textfile'
    if os.path.exists(path):  # Check if the file exists
        with open(path, 'r+') as file:
            content = file.read()
            if 'Rebooted' in content:  # Check if the Rebooted word is in the file content
                file.truncate(0)
                return True
            else:
                return False
    else:
        return False

if __name__ == "__main__":
    just_rebooted_must_wait = just_rebooted()
    if not just_rebooted_must_wait:
        main()
