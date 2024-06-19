"""
        Capture and store images to file, rapidly and efficiently. 
        It is designed to be run on a Raspberry Pi with a camera V2 module. 
        If no arguments are given, it will take two images and save them to /pi/images folder, then exit.
        Beforehand, create the default output folder with 'sudo mkdir /home/pi/images' and 'sudo chmod 777 /home/pi/images'.
"""
import os
import argparse
import os
import cv2
from PIL import Image
import time
from utils.camera_handler import CameraHandler
from fractions import Fraction
from google_storage import upload_blob
from io import BytesIO
import sys

# All imports MUST be above the following import and GreenGrass installation
sys.path.append('/home/pi/config')
from config import API_ENDPOINT, API_KEY, SERIAL, TOKEN

def upload_image(img, image_filename: str):
    # convert BGR to RGB and right format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    bytesIO = BytesIO()
    img_pil.save(bytesIO, format='JPEG')
    try:
        upload_blob("hallmonitor_3_input", image_filename, bytesIO)
    except Exception as e:
        print(e)


def filename_with_args(args, time_of_capture, resolution_tuple, awb_gains_tuple):
    filename = f"{time_of_capture}_res{resolution_tuple}_br{args.brightness}_co{args.contrast}_iso{args.iso}_awb_gains{awb_gains_tuple}_awb_mode{args.awb_mode}_exp{args.exposure_mode}_ss{args.shutter_speed}_mm{args.meter_mode}.jpg"
    return filename

def store_image(image, image_filename:str, output_folder:str, save_quality:int):
    # Convert BGR to RGB and right format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    filepath = os.path.join(output_folder, image_filename)
    image_pil.save(filepath, quality=save_quality, optimize=True)
    return 0



def create_output_folder(output_folder):
    try:
        if not os.path.exists(output_folder):
            os.system(f"sudo mkdir {output_folder}")
            os.system(f"sudo chmod 777 {output_folder}")
        return output_folder
    except Exception as e: 
        print(f"You must manually 'sudo mkdir {output_folder}' and 'sudo chmod 777 {output_folder}' due to error: {e}")
        exit(1)


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


def main(args):
    seconds_per_image = 6.3
    output_folder = create_output_folder(args.output_folder)
    min_to_complete, sec_to_complete = divmod((args.n_images * seconds_per_image)+5, 60) # 5 sec cam start up
    resolution_tuple = (args.resolution1, args.resolution2)
    awb_gains_tuple = (Fraction(args.awb_gains_red), Fraction(args.awb_gains_blue))
    framerate_fraction = Fraction(1, args.framerate)
    time_start_loop = time.time()
    print(f"\nStarting capture of {args.n_images} images. This will take approximately {min_to_complete:.0f}min {sec_to_complete:.0f}sec. Quality: {args.save_quality}/100.")
    try:
        camerahandler = CameraHandler(
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
        
        print(camerahandler.get_configurations())
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    try:
        for i in range(args.n_images):
            image, time_of_capture = camerahandler.capture_image()
            image_filename = f"{str(time_of_capture)}-awb{float(camerahandler.picamera.awb_gains[0]):.2f}_{float(camerahandler.picamera.awb_gains[1]):.2f}_br{camerahandler.picamera.brightness}_co{camerahandler.picamera.contrast}_ec{camerahandler.picamera.exposure_compensation}_iso{camerahandler.picamera.iso}_{camerahandler.picamera.resolution}_sm{camerahandler.picamera.sensor_mode}_ss{camerahandler.picamera.shutter_speed}_sq{args.save_quality}_{camerahandler.picamera.exposure_speed}_fr{args.framerate}.jpg"
            store_image(image, image_filename, output_folder, args.save_quality)
            if(i == 0):
                print(f"Filename: {image_filename}")
            elif(i%10 == 0):
                print(f"Images captured: {i}/{args.n_images}. Time passed: {time.time() - time_start_loop:.2f}sec.\n")
            else:
                print(".", end="", flush=True)
            if args.upload_image:
                print("Uploading image to cloud storage.")
                upload_image(image, image_filename)
    except Exception as e:
        print(e)
    finally:
        # Ensure the picamera is closed even if an exception is raised.
        min_completed, sec_completed = divmod((time.time()-time_start_loop), 60)
        print(f"\nCapture of {args.n_images} images complete in {min_completed:.0f}min {sec_completed:.0f}sec. Per image: {(time.time() - time_start_loop)/args.n_images:.2f}sec. Images saved to '{output_folder}. Last filename: {image_filename}'.")
        camerahandler.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)