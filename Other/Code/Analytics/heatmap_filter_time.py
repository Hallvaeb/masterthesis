import supervision as sv
import cv2
import numpy as np
import pandas as pd
from logs_to_df import read_files_to_dataframe
import os

# Config threshold for detections
CONFIDENCE_THRESHOLD = 0.5

# Paths
LOGS_DIR_PATH = './logs/'
OUTPUT_PATH = './supervision_heatmap.jpg' 
TARGET_DIR_PATH = './heatmaps_time'
base_image_path = './heatmap_base.jpg'

# Get the detections
df = read_files_to_dataframe(LOGS_DIR_PATH)

# Convert date and time_of_detection to datetime objects for easier filtering
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time_of_detection'], dayfirst=True)
df['time'] = df['datetime'].dt.time

# Filter out low confidence detections
df_filtered = df[df['confidence'] > CONFIDENCE_THRESHOLD]

# Function to filter detections by time range
def filter_detections_by_time(df, start_time, end_time):
	start_time = pd.to_datetime(start_time).time()
	end_time = pd.to_datetime(end_time).time()
	if start_time < end_time:
		return df[(df['time'] >= start_time) & (df['time'] < end_time)]
	else:
		# This handles the case where the interval wraps around midnight
		return df[(df['time'] >= start_time) | (df['time'] < end_time)]

# Define time intervals
time_intervals = [
	('06:00', '10:00'),
	('10:00', '11:00'),
	('11:00', '12:00'),
	('12:00', '13:00'),
	('13:00', '14:00'),
	('14:00', '15:00'),
	('15:00', '16:00'),
	('16:00', '17:00'),
	('17:00', '23:00'),
]

# Ensure the target directory exists
if not os.path.exists(TARGET_DIR_PATH):
	os.makedirs(TARGET_DIR_PATH)

n_detections_total = len(df_filtered) # Population size

# Load the base image
base_image = cv2.imread(base_image_path)

# Generate heatmaps for the specified time intervals
for start_time, end_time in time_intervals:
	df_time_filtered = filter_detections_by_time(df_filtered, start_time, end_time)
	print(f'Detections for {start_time}-{end_time}: {len(df_time_filtered)}')
	if len(df_time_filtered) > 0:
		detections = sv.Detections(
			xyxy=df_time_filtered[['x1', 'y1', 'x2', 'y2']].values,
			confidence=df_time_filtered['confidence'].values,
			class_id=np.zeros(len(df_time_filtered), dtype=int),
		)
		
		heat_map_annotator = sv.HeatMapAnnotator(
			position = sv.Position.CENTER,
			opacity = 0.5,
			radius = 30,
			kernel_size = 50,
			top_hue = 0,
			low_hue = 200,
		)
		
		annotated_image = heat_map_annotator.annotate(
			scene = base_image.copy(),
			detections = detections
		)

		# Add population and sample size to the image
		n_detections = len(df_time_filtered) # Sample size
		text = f'n={n_detections}, N={n_detections_total}'
		font = cv2.FONT_HERSHEY_SIMPLEX
		font_scale = 1
		color = (255, 255, 255)  # White color
		thickness = 2
		text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
		text_x = annotated_image.shape[1] - text_size[0] - 10  # 10 pixels from right edge
		text_y = annotated_image.shape[0] - 10  # 10 pixels from bottom edge
		cv2.putText(annotated_image, text, (text_x, text_y), font, font_scale, color, thickness)

		image_name = f'heatmap_time_{start_time.replace(":", "")}_{end_time.replace(":", "")}.jpg'
		with sv.ImageSink(target_dir_path = TARGET_DIR_PATH, overwrite=False) as sink:
			sink.save_image(
				image = annotated_image,
				image_name = image_name
			)

print("Heatmaps generated for specified time intervals.")
