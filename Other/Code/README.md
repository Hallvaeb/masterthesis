
CAMERA FOLDER:
capture_detect was previously named main.py, and was used to capture an image, detect (localize) persons in the image using yolov3 (weights must be downloaded), store the locations in a txt file (see Utils below), and send a blurred image to another google bucket.

capture_store was used to collect the images for the dataset. It simply captures images and stores them in a folder. 

Utils:
In Utils/ is the scripts used for creating a daily log file (which the detections was stored within) and sending this daily log. The code is to perform the inference, to setup the camera, to instantiate the logger class, and to upload the log. 


UTILS FOLDER:
Some of the code in utils/ is mentioned in the thesis, and was mainly used to convert data between different formats.


ANALYTICS:
Here are the heatmap generation code. Note that the code is not optimized. The best template for creating heatmaps would be the simple heatmap.py file.
logs_to_df.py reveals the basic dumb way that was used to read data from log files to a dataframe.

The daily txt log file looks like this:
09:10:06: 00.00s -------------START--------------
09:10:37: 31.12s 0 detections made.
09:10:52: 46.28s File 'hallvard-210224-091037-0.jpg' uploaded.
09:10:55: 48.68s -------------FINISH-------------
09:20:05: 00.00s -------------START--------------
09:20:37: 31.10s 3 detections made.
09:20:51: 45.47s 
Detection 1: Confidence: 0.96. Box coords: [380, 145, 422, 299].
Detection 2: Confidence: 0.83. Box coords: [319, 143, 352, 217].
Detection 3: Confidence: 0.36. Box coords: [575, 90, 631, 186].
09:20:52: 46.66s File 'hallvard-210224-092037-3.jpg' uploaded.
09:20:55: 49.15s -------------FINISH-------------





If further explanations are needed, contact hallvard.bjorgen@gmail.com