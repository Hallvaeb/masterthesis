"""trt_yolo.py
This script uses TensorRT optimized YOLO engine to detect people on images
"""

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from yolov3.utils import detect_image, load_yolo_model, create_and_save_yolo_model


class Detection:
    def __init__(self, box, conf, cl):
        self.box = box
        self.conf = conf
        self.cl = cl

    def __str__(self):
        return f"Box coordinates: {str(self.box)}, confidence: {str(self.conf)}\n"

def detect_persons(numpyImage, trt_yolo, conf_th):
    """
    @param numpyImage: image as numpy array
    @param trt_yolo: instance of TrtYolo class
    @param conf_th: Confidence threshold
    @return: list of detections (boxes, class, confidence)
    """
    # boxes, confs, clss = trt_yolo.detect(numpyImage, conf_th)
    boxes, confs, clss = detect_image(Yolo=trt_yolo, numpyImage=numpyImage, score_threshold=conf_th, output_path='')
    
    # HallMonitor version
    person_boxes = []
    clss_filtered = []
    conf_filtered = []
    for i in range(len(boxes)):
        if clss[i] == 0:
            person_boxes.append((boxes[i]))
            clss_filtered.append(clss[i])
            conf_filtered.append(confs[i])

    return person_boxes, clss_filtered, conf_filtered


def detect(img, trt_yolo, conf_th):
    """
    @param img: image captured already
    @param trt_yolo: TrtYolo instance
    @param conf_th: Confidence threshold
    @param vis: Visualization instance
    """
    # detect on full image
    person_boxes_full, clss_filtered_full, conf_filtered_full = detect_persons(
        img, trt_yolo, conf_th
    )

    detections = wrap_detections_values(
        person_boxes_full, conf_filtered_full, clss_filtered_full
    )

    return detections    

def wrap_detections_values(boxes, confs, clss):
    detections = []
    for i in range(len(boxes)):
        detection = Detection(boxes[i], confs[i], clss[i])
        detections.append(detection)
    return detections

def yolov3_inference(image, conf_threshold=0.25, model_path = '/home/pi/ML_models/models/yolo_model_hallmonitor.h5'):
    # Schedule the model loading and image capturing to run concurrently
    if os.path.exists(model_path):
        yolov3_model = load_yolo_model(model_path)
    else:
        print(f"Model file does not exist. Creating model and saving to {model_path}.")
        # Here you can create your model or handle the absence of the model file appropriately
        yolov3_model = create_and_save_yolo_model(model_path)

    # Detect persons
    detections = detect(image, yolov3_model, conf_threshold)

    return detections