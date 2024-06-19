"""
This file will convert daily log files to a pandas DataFrame. The DataFrame will have the following columns:
- date: The date of the log file.
- time_of_detection: The time of the detection.
- confidence: The confidence of the detection.
- x1: The x-coordinate of the top-left corner of the bounding box.
- y1: The y-coordinate of the top-left corner of the bounding box.
- x2: The x-coordinate of the bottom-right corner of the bounding box.
- y2: The y-coordinate of the bottom-right corner of the bounding box.
- file_name: The name of the file where the detection was made.

daily txt log file looks like this:
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

Explanation:
Detection 1: Confidence: CONFIDENCE_VALUE. Box coords: [x1, y1, x2, y2].
"""
import pandas as pd
import os
from tqdm import tqdm

def read_files_to_dataframe(logs_folder='./logs'):
	df = pd.DataFrame(columns=['date', 'time_of_detection', 'confidence', 'x1', 'y1', 'x2', 'y2', 'file_name', 'start_time', 'end_time'])
	for file in tqdm(os.listdir(logs_folder)):
		file_serial_no = file.split("-")[0]
		if (file.endswith(".txt")):
			date = "-".join(file.split('.')[0].split("-")[1:])
			detections= []
			with open(logs_folder+'/'+file, 'r') as f:
				lines = f.readlines()
				for line in lines:
					try:
						if 'START' in line:
							start_time = line[:8]
						if 'detections' in line:
							time_of_detection = (line[:8])
						if 'Detection' in line:
							confidence = line.split()[3][:-1]
							box_coords = line.split("[")[1].split("]")[0]
							box_coords = box_coords.split(", ")
							box_coords = [round(float(b), 4) for b in box_coords]
							detections.append((confidence, box_coords))
						if 'uploaded' in line:
							file_name = line.split("'")[1]
						if 'FINISH' in line:
							end_time = line[:8]
							for det in detections:
								df.loc[len(df.index)] = [date, time_of_detection, det[0], det[1][0], det[1][1], det[1][2], det[1][3], file_name, start_time, end_time]
							detections = []
					except:
						continue
	df['confidence'] = df['confidence'].astype(float)
	return df

