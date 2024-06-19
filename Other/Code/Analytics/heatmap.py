import supervision as sv
import cv2
import numpy as np
from logs_to_df import read_files_to_dataframe
import os

# Config threshold for detections
CONFIDENCE_THRESHOLD = 0.5

# Paths
LOGS_DIR_PATH = './logs_selection'
TARGET_DIR_PATH = './heatmaps'
TARGET_FILE_NAME = 'heatmap_bottom_center.jpg'
BASE_IMAGE_PATH = './heatmap_base_right.jpg'

# Ensure the target directory exists
if not os.path.exists(TARGET_DIR_PATH):
    os.makedirs(TARGET_DIR_PATH)

# Heatmap config
heat_map_annotator = sv.HeatMapAnnotator(
    position    = sv.Position.BOTTOM_CENTER, # BOTTOM_CENTER
    opacity     = 0.5,
    radius      = 40,   # Smaller radius for finer details
    kernel_size = 35,   # Smaller kernel size for finer details
    top_hue     = 0,    # Red for high density
    low_hue     = 240,  # Blue for low density
)

# Get the detections
df = read_files_to_dataframe(logs_folder = LOGS_DIR_PATH)

# Filter out low confidence detections
df_filtered = df[df['confidence'] > CONFIDENCE_THRESHOLD]

# Create sv.Detections object
detections = sv.Detections(
    xyxy=df_filtered[['x1', 'y1', 'x2', 'y2']].values,
    confidence=df_filtered['confidence'].values,
    class_id=np.zeros(len(df_filtered), dtype=int),
)

# Load the image
base_image = cv2.imread(BASE_IMAGE_PATH)

# Draw heatmap
annotated_image = heat_map_annotator.annotate(
    scene = base_image.copy(),
    detections = detections
)

# Add population to the image
text = f'n={len(df_filtered)}'
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 5
color = (255, 255, 255)  # White color
thickness = 4
text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
text_x = annotated_image.shape[1] - text_size[0] - 10  # 10 pixels from right edge
text_y = annotated_image.shape[0] - 10  # 10 pixels from bottom edge    
cv2.putText(annotated_image, text, (text_x, text_y), font, font_scale, color, thickness)

# Save the heatmap image
with sv.ImageSink(target_dir_path = TARGET_DIR_PATH, overwrite=False) as sink: # Overwrite affects whole folder
    sink.save_image(
        image = annotated_image,
        image_name = TARGET_FILE_NAME
    )

print(f"Heatmap saved to {os.path.join(TARGET_DIR_PATH, TARGET_FILE_NAME)}")
