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
TARGET_DIR_PATH = './heatmaps_time_and_date_intervals'
base_image_path = './heatmap_base.jpg'

# Get the detections
df = read_files_to_dataframe(LOGS_DIR_PATH)

# Convert date and time_of_detection to datetime objects for easier filtering
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time_of_detection'], dayfirst=True)

# Filter out low confidence detections
df_filtered = df[df['confidence'] > CONFIDENCE_THRESHOLD]

# Function to filter detections by date and time range
def filter_detections(df, start_datetime=None, end_datetime=None):
    if start_datetime:
        df = df[df['datetime'] >= start_datetime]
    if end_datetime:
        df = df[df['datetime'] <= end_datetime]
    return df

# Example: Filter by specific date and time range
# start_datetime = pd.to_datetime('2024-05-06 09:00:00')
# end_datetime = pd.to_datetime('2024-05-06 12:00:00')
# df_filtered = filter_detections(df_filtered, start_datetime, end_datetime)

# Create sv.Detections object
detections = sv.Detections(
    xyxy=df_filtered[['x1', 'y1', 'x2', 'y2']].values,
    confidence=df_filtered['confidence'].values,
    class_id=np.zeros(len(df_filtered), dtype=int),
)

# Load the image
base_image = cv2.imread(base_image_path)

# Save heatmap image
if not os.path.exists(TARGET_DIR_PATH):
    os.makedirs(TARGET_DIR_PATH)


# Generate heatmaps for different times of the day
def generate_time_based_heatmaps(df, base_image, interval='D', target=TARGET_DIR_PATH):
    time_ranges = pd.date_range(start=f'{df['date'].min()} 06:00', end=f'{df['datetime'].max()} 06:00', freq=interval)
    n_detections_total = len(df) # Population size
    
    for i in range(len(time_ranges) - 1):
        start_time = time_ranges[i]
        end_time = time_ranges[i + 1]
        
        df_time_filtered = filter_detections(df, start_datetime=start_time, end_datetime=end_time)
        
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
            

            if not os.path.exists(target):
                os.makedirs(target)

            # Save the heatmap image
            if(interval == 'D'):
                image_name = f'heatmap_day_{start_time.strftime("%d%m%Y")}.jpg'
            elif(interval == 'W'):
                week_number= start_time.isocalendar()[1]
                image_name = f'heatmap_week_number_{week_number}_{start_time.strftime("%Y")}.jpg'
            elif(interval == 'ME'):
                image_name = f'heatmap_month_{start_time.strftime("%m%Y")}.jpg'
            elif start_time.date() != end_time.date():
                image_name = f'heatmap_from_{start_time.strftime("%d%m%Y_%H%M")}_to_{end_time.strftime("%H%M")}.jpg'
            else: 
                image_name = f'heatmap_from_{start_time.strftime("%d%m%Y_%H%M")}_to_{end_time.strftime("%d%m%Y_%H%M")}.jpg'
            with sv.ImageSink(target_dir_path = target, overwrite=False) as sink:
                sink.save_image(
                    image = annotated_image,
                    image_name = image_name
                )

# Example usage: Generate heatmaps with 1-hour intervals
generate_time_based_heatmaps(df_filtered, base_image, interval='D', target=TARGET_DIR_PATH+ '/daily')
generate_time_based_heatmaps(df_filtered, base_image, interval='W', target=TARGET_DIR_PATH+ '/weekly')
generate_time_based_heatmaps(df_filtered, base_image, interval='ME', target=TARGET_DIR_PATH+ '/monthly')
