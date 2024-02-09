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

from coordinate_helper import convert_coordinates, compute_coordinates_of_shrunk_image
# from upload.dropbox_helper import upload_image
from upload.upload_schedule_reader import should_upload_image_timebased, should_upload_image
from upload.google_storage import upload_blob
from image_manipulator import gamma_correction, split_image, draw_text_on_image
from rectangle_math import (
	remove_boxes_inside_main,
	merge_boxes_belonging_to_the_same_person,
	check_for_splitsize_boxes_in_flat_list
)
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
		"YOLO model on Jetson"
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
	parser.add_argument(
		"-x",
		"--xtiles",
		type=int,
		default=5,
		help="Amount of tiles to split image horizontally for analyzing with better accuracy",
	)
	parser.add_argument(
		"-y",
		"--ytiles",
		type=int,
		default=3,
		help="Amount of tiles to split image vertically for analyzing with better accuracy",
	)
	return parser.parse_args()


def push_image(img, amountOfPeople: int, timestamp):
	"""
	Uploads image to bucket. Sends count to endpoint
	@param img:image as numpy array
	@param amountOfPeople: amount of detected persons in the image
	"""
	fileName = "NULL"
	should_upload = False

	try:
		should_upload = should_upload_image()  # schedule based
	except:
		global current_minutes
		write_to_log("caught exception while trying to use upload schedule")
		should_upload = should_upload_image_timebased(current_minutes)

	if should_upload:
		write_to_log("Uploading image to cloud")
		fileName = str(uuid.uuid4()) + ".jpg"
		# convert BGR to RGB
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		img_pil = Image.fromarray(img)
		buffered2 = BytesIO()
		img_pil.save(buffered2, format="JPEG")

		try:
			upload_blob("hallmonitor_3_input", fileName, buffered2)
		except Exception as e:
			write_to_log("Failed to upload image to cloud: " + e, 1)

	DEVICE_ID = hashlib.sha256((SERIAL + TOKEN).encode('utf-8')).hexdigest()
	print("Device ID: ", DEVICE_ID)
	data = {
		"key": API_KEY,
		"file_name": fileName,
		"count": amountOfPeople,
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
        print(detections)
        data = {"device_id": DEVICE_ID, "file": filename, "detections": detections, "created_at": timestamp}

        http = requests.Session()
        response = http.post(url=ngrok_test_ip, json=data, headers=headers)
        print(response.status_code)


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
	print("\nDetecting objects..")
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

def take_picture(cam, trt_yolo, conf_th, vis, horizontal_tile_amount, vertical_tile_amount):
	"""
	@param cam: Camera instance
	@param trt_yolo: TrtYolo instance
	@param conf_th: Confidence threshold
	@param vis: Visualization instance
	"""
	print("Taking picture..")
	write_to_log("Capturing image")
	timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
	timestamp_draw = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

	img = cam
	img = gamma_correction(img)
	img = cv2.GaussianBlur(img, (5, 5), 0)
	write_to_log("Image preprocessing")

	if img is not None:
		detections_3d = []
		detections_flat = []

		# detection on full image
		write_to_log("Running detection on full image", True)

		person_boxes_full, clss_filtered_full, conf_filtered_full = detect_persons(
			img, trt_yolo, conf_th
		)

		detections_full_img = wrap_detections_values(
			person_boxes_full, conf_filtered_full, clss_filtered_full
		)

		overlay_percentage = 0
		write_to_log("Splitting image")
		images = split_image(
			img, horizontal_tile_amount, vertical_tile_amount, overlay_percentage
		)

		write_to_log("Running detection on splitted images")
		for vertical_pos in range(horizontal_tile_amount):
			detections_in_column = []
			for horizontal_pos in range(vertical_tile_amount):
				person_boxes, clss_filtered, conf_filtered = detect_persons(
					images[horizontal_pos][vertical_pos], trt_yolo, conf_th
				)

				detections_in_tile = []

				for k in range(len(person_boxes)):
					print("Converting")
					x, y = convert_coordinates(
						person_boxes[k][2],
						person_boxes[k][3],
						overlay_percentage,
						vertical_pos,
						horizontal_pos,
						img.shape[0],
						img.shape[1],
						horizontal_tile_amount,
						vertical_tile_amount,
					)
					x2, y2 = convert_coordinates(
						person_boxes[k][0],
						person_boxes[k][1],
						overlay_percentage,
						vertical_pos,
						horizontal_pos,
						img.shape[0],
						img.shape[1],
						horizontal_tile_amount,
						vertical_tile_amount,
					)
					rect = [x2, y2, x, y]
					detection = Detection(rect, conf_filtered[k], clss_filtered[k])
					# detections_flat.append(detection)
					detections_in_tile.append(detection)

				detections_in_column.append(detections_in_tile)

			detections_3d.append(detections_in_column)

		detections_merged = merge_boxes_belonging_to_the_same_person(
			detections_3d,
			img.shape[1],
			img.shape[0],
			horizontal_tile_amount,
			vertical_tile_amount,
			5,
		)
		remove_threshhold = 6  # percent. smaller value means more strict
		removed_detections = check_for_splitsize_boxes_in_flat_list(detections_merged, images, remove_threshhold)
		for i in removed_detections:
			print('Rem: ' + str(i.box))

		remove_boxes_inside_main(detections_full_img, detections_merged)

		# BLURRED
		write_to_log("Blurring image")
		img = cv2.GaussianBlur(img, (211, 211), 0)

		# remove boxes inside main boxes
		# remove_boxes_inside_main(detections_full_img, detections_flat)
		# remove_boxes_inside_other_boxes(detections_flat)

		divide_amount = 4
		img = upload_preprocess(img, divide_amount)

		write_to_log("Combining detections")

		compute_coordinates_of_shrunk_image(detections_full_img, divide_amount)
		compute_coordinates_of_shrunk_image(detections_merged, divide_amount)
		compute_coordinates_of_shrunk_image(removed_detections, divide_amount)

		# img = vis.draw_bboxes(img, detections_flat, 0)
		write_to_log("Visualizing detections", True)
		img = vis.draw_bboxes(img, detections_full_img, 1)
		img = vis.draw_bboxes(img, detections_merged, 1)
  
		"""Debugging purposes for pushing detection coordinates"""
		print("Detections on full image: ", detections_full_img)
		aNumber = 0
		cNumber = 0

		all_detections = detections_full_img + detections_merged
		for detection in detections_full_img:
				# print(aNumber, "Detection (Full-image): ", detection)
				print("Confidence: ", detection.conf)
				print("Box coords: ", detection.box)
				print("Class: ", detection.cl)
				aNumber = aNumber + 1
		for detection in detections_merged:
				# print(cNumber, "Detection (Merged): ", detection)
				print("Confidence: ", detection.conf)
				print("Box coords: ", detection.box)
				print("Class: ", detection.cl)
				cNumber = cNumber + 1
		print("Detections on merged image: ", detections_merged)

		# img = vis.draw_bboxes(img, removed_detections, 6) # draws removed (tile-size) detections
		# for testing purposes
		count = len(detections_merged) + len(detections_full_img)
		# draw_people_count(img, count)
		# timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
		# timestamp_draw = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
		fileName = f"{str(uuid.uuid4())}.jpg"
  
		draw_text_on_image(img, timestamp_draw)
		push_image(img, count, timestamp)  # push to api using post request
		if ZONE_ANALYSIS == "TRUE":
  			upload_detection_data(fileName, all_detections, timestamp)
		# upload_image_dropbox(img, SERIAL_NR) #upload to dropbox for testing


def wrap_detections_values(boxes, confs, clss):
	detections = []
	for i in range(len(boxes)):
		detection = Detection(boxes[i], confs[i], clss[i])
		detections.append(detection)
	return detections


# this is only used for testing
def upload_image_dropbox(img_array, path):
	"""
	@param img_array:
	@param path:
	"""
	img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
	img_pil = Image.fromarray(img_array)
	buffered = BytesIO()
	img_pil.save(buffered, format="JPEG")
	# img_pil.save(buffered, optimize=True, quality=50, format="JPEG")
	upload_image(buffered, path)


# (--type 0) default. analyzes one picture and exits script
def start(cam, vis, trt_yolo, conf_th, horizontal_tile_amount, vertical_tile_amount):
	take_picture(cam, trt_yolo, conf_th, vis, horizontal_tile_amount, vertical_tile_amount)


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


def stop_camera(cam):
	try:
		write_to_log("Stopping Camera..")
		cam.release()
		write_to_log("Camera stopped", True)
	except Exception as e:
		write_to_log("Camera is not getting released: " + str(e.__doc__), True)


# shrinks image img (numpy array) based on divide_amount value
def upload_preprocess(img, divide_amount):
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
	write_to_log("====== STARTING ======", True)
 
	process_check.processKiller()

	now = datetime.now()

	max_value = 1  # a value of 1 means it has to fail twice to attempt reboot. 0 is initial value.
	fail_count = failure_count.get_value()  # if the script failed a bunch of times in a row it
	if fail_count > 0:
		write_to_log("Failed " + str(fail_count) + " time(s)", True)

	if fail_count > max_value:  # should reboot. e.g reboot clears memory
		try:
			failure_count.reset()
			write_to_log('(attempting REBOOT on start)', True)
		except:
			pass
		os.system('sudo reboot')

	global current_minutes
	current_minutes = int(now.strftime("%M"))
	args = parse_args()
	camera_rot = args.cam_flip
	print(f"Camera rot: {camera_rot}")
	conf_th = args.confidence / 100.0
	horizontal_tile_amount = args.xtiles
	vertical_tile_amount = args.ytiles

	trt_yolo = Load_Yolo_model()
	# trt_yolo = TrtYOLO(args.model, dimension, args.category_num)
	cls_dict = get_cls_dict(args.category_num)

	cam = None

	try:
		online = True

		if online:
			# cam = initiate_camera(args)
			cam = capture_image_pi(camera_rot)
			vis = BBoxVisualization(cls_dict)
			operation_mode(args, cam, vis, trt_yolo, conf_th, horizontal_tile_amount, vertical_tile_amount)
			# stop_camera(cam)
		else:
			write_to_log("Connection failed..", True)
			raise Exception("Connection not established")

	except Exception as e:
		print(" (Caught error) ")
		write_to_log('Caught error. Stopping cam\n' + str(e.__class__) + 'occured\n', True)
		write_to_log(traceback.format_exc(), True)
		if cam:
			stop_camera(cam)
		write_to_log("ShouldReboot? -> Count: " + str(failure_count.get_value()), True)
		should_reboot = failure_count.increment(max_value)
		if should_reboot:
			write_to_log('attempting REBOOT', True)
			os.system('sudo reboot')  # if there is close to no available ram this will fail
			# therefore a reboot check is also made at the start of
			# this script.

		raise Exception(e)

	failure_count.reset()
	write_to_log("====== STOPPING ======", True)


def checkForNetwork(tries):
	host = '8.8.8.8'

	delay = 0.5
	backoff = 1.5
	command = ['ping', '-c', '1', host]

	for n in range(tries):
		try:
			result = subprocess.call(command)
			if result == 2:
				write_to_log("Retrying.. in {0}".format(delay), True)
				time.sleep(delay)
				delay *= backoff
			elif result == 0:
				write_to_log("Connection established after {0} retries".format(n), True)
				return result == 0
		except:
			write_to_log("Connection not established after {0}".format(tries), True)


class Detection:
	def __init__(self, box, conf, cl):
		self.box = box
		self.conf = conf
		self.cl = cl

	def __str__(self):
		return str(self.box) + "\n" + str(self.conf) + "\n" + str(self.cl)

def scheduledRebootCheck():
    #This func returns true if detect should run.
    #It returns false if the device has just been rebooted at the scheduled time.
    path = '/home/ggc_user/reboot_at_night_textfile'
    if os.path.exists(path):  # Check if the file exists
        with open(path, 'r+') as file:
            content = file.read()
            if 'Rebooted' in content:  # Check if the Rebooted word is in the file content
                file.truncate(0)
                return False
            else:
                return True
    else:
        return True

if __name__ == "__main__":
	rebootBool = scheduledRebootCheck()
	if rebootBool:
		main()
