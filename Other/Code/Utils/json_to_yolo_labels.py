import json
import os

# Usage
# json_file_path = '/home/hallvaeb/data-masteroppgave/2nd-iteration/complete-annotations/annotations-json-minimum.json' 
json_file_path = '/home/hallvaeb/data-masteroppgave/in-progress/1st-iteration/predictions_failed_not_yet_labeled.json' 
# output_yolo_folder = '/home/hallvaeb/data-masteroppgave/2nd-iteration/complete-annotations/labels'
output_yolo_folder = '/home/hallvaeb/data-masteroppgave/in-progress/1st-iteration/labeled_images'

def json_to_yolo(json_file, output_folder):
    # Load JSON data
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    lines_written = 0
    
    # Process each entry in the JSON
    for item in data:
        try:
            filename_base = item['filename']
        except:
            print("Didn't find filename for item:", item)    
        output_path = os.path.join(output_folder, f"{filename_base}.txt")
        # Each label entry
        try:
            for label in item['label']:
                # Extract label details
                x = float(label['x'])
                y = float(label['y'])
                width = float(label['width'])
                height = float(label['height'])

                # Convert pixel coordinates to normalized YOLO format
                x_center_norm = (x + width / 2) / 100
                y_center_norm = (y + height / 2) / 100
                width_norm = width / 100
                height_norm = height / 100
                
                with open(output_path, 'a') as output_file:
                    lines_written += 1
                    # Write to file
                    output_file.write(f"0 {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n")
        except:
            print(filename_base, 'has no labels.')
    print(f'{lines_written} lines written.')

json_to_yolo(json_file_path, output_yolo_folder)
