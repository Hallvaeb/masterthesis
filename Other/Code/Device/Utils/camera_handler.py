import time
import picamera
import numpy as np
import cv2
from datetime import datetime
from fractions import Fraction


class CameraHandler:
    def __init__(self, awb_gains, awb_mode, brightness, contrast, exposure_compensation, exposure_mode, image_format, iso, meter_mode, resolution, sensor_mode, shutter_speed, framerate):
        # Create camera with the arguments
        cam = picamera.PiCamera(resolution=resolution, sensor_mode=sensor_mode, framerate=framerate) 
        
        # 100, 200, 320, 400, 500, 640, 800. 0 is auto. 
        # Test all values
        cam.iso = iso

        # Wait for the automatic gain control to settle
        time.sleep(2)
        
        # Shutter speed for static images, instead of framerate. In microseconds.
        # 0 to 6000000. Default 0. 0 is auto. Max 6s.
        cam.shutter_speed = shutter_speed

        # 'off' 'auto' 'night' 'nightpreview' 'backlight' 'spotlight' 'sports' 'snow' 'beach' 'verylong' 'fixedfps' 'antishake' 'fireworks'
        cam.exposure_mode = exposure_mode

        # Auto white balance. Default auto.
        # 'off' 'auto' 'sunlight' 'cloudy' 'shade' 'tungsten' 'fluorescent' 'incandescent' 'flash' 'horizon'
        cam.awb_mode = awb_mode

        # Set as a (red, blue) tuple. 0.0 to 8.0. Typical is 0.9-1.9. Only effect when awb_mode is 'off'. IMPORTANT: awb and exposure mode must be set to off _before_!
        cam.awb_gains = awb_gains
        
        # -25 to 25. Default 0. Adjusts the exposure compensation value. Each int = 1/6 stop.
        cam.exposure_compensation = exposure_compensation

        # Postprosessering, default is 50
        # When set, the property adjusts the brightness of the camera.
        cam.brightness = brightness

        # When set, the property adjusts the contrast of the camera. 
        # -100 to 100. Default 0.
        cam.contrast = contrast

        # 'bgr' 
        self.image_format = image_format

        # default is 'average'. Used by camera to determine exposure.
        # 'average' 'spot' 'backlit' 'matrix'. backlit largest area.
        cam.meter_mode = meter_mode
        self.picamera = cam


    def capture_image(self): # Rounds to the nearest multiple of 32.
        image = np.empty((self.picamera.resolution[1] * self.picamera.resolution[0] * 3,), dtype=np.uint8)
        self.picamera.capture(image, self.image_format)
        time_of_capture = datetime.now().strftime("%d%m%y-%H%M%S")
        image = image.reshape((self.picamera.resolution[1], self.picamera.resolution[0], 3))
        return image, time_of_capture

    @staticmethod
    def shrink_image(image, divide_amount):
        new_width = int(round(image.shape[1] / divide_amount))
        new_height = int(round(image.shape[0] / divide_amount))
        image = cv2.resize(image, dsize=(new_width, new_height), interpolation=cv2.INTER_NEAREST)
        return image
    
    def get_configurations(self):
        return f'''Camera configurations:
            analog_gain             = {self.picamera.analog_gain}
            awb_mode                = {self.picamera.awb_mode}
            awb_gains               = {float(self.picamera.awb_gains[0])}_{float(self.picamera.awb_gains[1])}
            brightness              = {self.picamera.brightness}
            contrast                = {self.picamera.contrast}
            digital_gain            = {self.picamera.digital_gain}
            exposure_compensation   = {self.picamera.exposure_compensation}
            exposure_speed          = {self.picamera.exposure_speed}
            exposure_mode           = {self.picamera.exposure_mode}
            framerate               = {self.picamera.framerate}
            iso                     = {self.picamera.iso}
            meter_mode              = {self.picamera.meter_mode}
            resolution              = {self.picamera.resolution}
            sensor_mode             = {self.picamera.sensor_mode}
            shutter_speed           = {self.picamera.shutter_speed}
            '''
    
    def close(self):
        self.picamera.close()