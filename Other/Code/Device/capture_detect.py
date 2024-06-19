"""trt_yolo.py
This script uses TensorRT optimized YOLO engine to detect people on images
"""

import os

# Set the PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION environment variable
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import sys

# LOAD
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import requests
from PIL import Image
import time
from datetime import datetime
from io import BytesIO
from utils.health_check import get_full_health_check
from utils.coordinate_helper import modify_coordinates_to_fit_shrunk_image

# from upload.dropbox_helper import upload_image
from upload.google_storage import upload_blob
from utils.image_manipulator import gamma_correction
from utils.camera import add_camera_args
from utils.visualization import draw_bboxes
from utils.logger_hallvard import LoggerHallvard
# import pycuda.autoinit  # This is needed for initializing CUDA driver
import hashlib
import os
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import utils.process_check
from utils.camera_handler import CameraHandler
from utils.yolov3_inference import yolov3_inference
import utils.failure_count
from fractions import Fraction

# All imports MUST be above the following import and GreenGrass installation
sys.path.append('/home/pi/config')
from config import API_ENDPOINT, API_KEY, SERIAL, TOKEN

WINDOW_NAME = "TrtYOLODemo"
dimension = (608, 608)
current_minutes = ""


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Capture settings
    parser.add_argument("-of","--output_folder",type=str,default="/home/pi/images",help="Folder to save images to.",)
    parser.add_argument("-ni","--n_images",type=int,default=1,help="How many images to capture.",)
    parser.add_argument("-sq","--save_quality", default=90, type=int,)
    
    # Camera handler settings
    parser.add_argument("-re1","--resolution1", default=3264, type=int,)
    parser.add_argument("-re2","--resolution2", default=2464, type=int,)
    parser.add_argument("-br","--brightness", default=55, type=int,)
    parser.add_argument("-co","--contrast", default=0, type=int,)
    parser.add_argument("-is","--iso", default=640, type=int,)
    parser.add_argument("-em","--exposure_mode", default='off', type=str,)
    parser.add_argument("-ec","--exposure_compensation", default=0, type=int,)
    parser.add_argument("-if","--image_format", default='bgr', type=str,)
    parser.add_argument("-ss","--shutter_speed", default=80000, type=int,)
    parser.add_argument("-am","--awb_mode", default='off', type=str,)
    parser.add_argument("-agr","--awb_gains_red", default=1.5, type=float,)
    parser.add_argument("-agb","--awb_gains_blue", default=1.5, type=float,)
    parser.add_argument("-mm","--meter_mode", default='average', type=str,)
    parser.add_argument("-fr","--framerate", default=6, type=int,)
    parser.add_argument("-sm","--sensor_mode", default=3, type=int,)
    parser.add_argument("-ui", "--upload_image", action="store_true", help="Upload image to cloud storage bucket.",)
    parser.parse_args()
    args = parser.parse_args()
    return args

def upload_image(img, image_filename: str):
    """
    Uploads image to bucket. Sends count to endpoint
    @param img:image as numpy array
    @param count: count of detected persons in the image
    """
    # convert BGR to RGB and right format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    buffered2 = BytesIO()
    img_pil.save(buffered2, format="JPEG")

    try:
        upload_blob("hallmonitor_3_input", image_filename, buffered2)
        log.write(f"File '{image_filename}' uploaded.")
    except Exception as e:
        log.write(f"Failed to upload image to cloud: {e}")


def upload_health_data():
    health_data = get_full_health_check()

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
        response = http.post(url=ENDPOINT, data=health_data)
        if response.status_code != 201:
            log.write(f"{str(response.status_code)}: Device health data not uploaded.")
    except Exception as e:
        log.write(f'Error uploading device health data. e: {e}')


def upload_analytic_results(image_annotated_filename, count, time_of_capture):

    # Upload count
    DEVICE_ID = hashlib.sha256((SERIAL + TOKEN).encode('utf-8')).hexdigest()
    data = {
        "key": API_KEY,
        "file_name": image_annotated_filename,
        "count": count,
        "device_id": DEVICE_ID,
        "timestamp": time_of_capture
    }

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
        if response.status_code != 201:
            log.write(f"{str(response.status_code)}: Analytic not uploaded.")
    except Exception as e:
        log.write(f'Error uploading analytic results. e: {e}')



# shrinks image img (numpy array) based on divide_amount value
def shrink_image(img, divide_amount):
    new_width = img.shape[1] / divide_amount
    new_height = img.shape[0] / divide_amount

    new_width_int = int(round(new_width))
    new_height_int = int(round(new_height))

    img = cv2.resize(
        img, dsize=(new_width_int, new_height_int), interpolation=cv2.INTER_NEAREST
    )
    return img


def get_camera_handler():
    # Kill already started detect processes
    utils.process_check.processKiller()

    # If the script failed a bunch of times in a row it should reboot to clear memory
    failure_count_value = utils.failure_count.get_value()  
    if failure_count_value > 1:
        log.write(f"Fail count was {failure_count_value}, rebooting system.")
        os.system('sudo reboot')

    resolution_tuple = (args.resolution1, args.resolution2)
    awb_gains_tuple = (Fraction(args.awb_gains_red), Fraction(args.awb_gains_blue))
    framerate_fraction = Fraction(1, args.framerate)
    time_start_loop = time.time()
    try:
        camera_handler = CameraHandler(
            awb_gains=awb_gains_tuple,
            awb_mode=args.awb_mode,
            brightness=args.brightness,
            contrast=args.contrast,
            exposure_compensation=args.exposure_compensation,
            exposure_mode=args.exposure_mode,
            image_format=args.image_format,
            iso=args.iso,
            meter_mode=args.meter_mode,
            resolution=resolution_tuple,
            sensor_mode=args.sensor_mode,
            shutter_speed=args.shutter_speed,
            framerate=framerate_fraction)
        
        print(camera_handler.get_configurations())
        return camera_handler
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

def just_rebooted_at_night():
    # This func returns true if device was just rebooted at night. In that case we it's unnessessary to run detect.
    path = '/home/ggc_user/reboot_at_night_textfile'
    if os.path.exists(path):  # Check if the file exists
        os.remove(path) # Delete if it has.
        return True
    else:
        return False


def main(args):
    camera_handler = get_camera_handler()
    image_captured, time_of_capture = camera_handler.capture_image()

    detections = yolov3_inference(image_captured)

    # Count is the lenght of detections
    count  = len(detections)
    log.write(f"{count} detections made.")

    # Create file name
    timestamp_annotated = datetime.now().strftime("%d%m%y-%H%M%S")
    image_annotated_filename = f"hallvard-{str(timestamp_annotated)}-{str(count)}.jpg"

    # Logging all detections at once is more efficient
    log_detections = [f"\nDetection {i+1}: Confidence: {det.conf:.2f}. Box coords: {det.box}." for i, det in enumerate(detections)]
    if log_detections:  # This checks if log_detections is not empty.
        log.write("".join(log_detections))

    if count>0:
        # Compress image, adjust detections
        shrink_denominator = 4
        log.write("Blurring...", False)
        image_annotated = cv2.GaussianBlur(image_captured, (191, 191), 0) # value for blur amount experimentally tested.
        log.write("Shrinking...", False)
        image_annotated = shrink_image(image_annotated, shrink_denominator)
        log.write("Computing coordinates...", False)
        modify_coordinates_to_fit_shrunk_image(detections, shrink_denominator)

        log.write("Drawing bounding boxes...", False)
        image_annotated = draw_bboxes(image_annotated, detections)

        log.write("Uploading image...", False)
        upload_image(image_annotated, image_annotated_filename)
        
        log.write("Uploading health data...", False)
        upload_health_data()


if __name__ == "__main__":
    just_rebooted_at_night = just_rebooted_at_night()
    if not just_rebooted_at_night:
        args = parse_args()
        start_time = time.time()
        log = LoggerHallvard(start_time)
        log.write("-------------START--------------")
        main(args)
        log.write("-------------FINISH-------------\n")