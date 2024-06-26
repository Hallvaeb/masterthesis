
import pandas as pd
import os
import numpy as np

def rename_files_in_directory(directory):
	# List all files in the directory
	for filename in os.listdir(directory):
		# Construct the full file path
		file_path = os.path.join(directory, filename)
		
		# Ensure we are working with a file
		if os.path.isfile(file_path):
			# Extract the first 13 characters for the new name
			new_name = filename[:13]
			
			# Check if 'hm12' is part of the filename to determine the suffix
			if 'hm12' in filename:
				new_name += "-left"
			else:
				new_name += "-right"
			
			# Get the file extension
			file_extension = os.path.splitext(filename)[1]
			
			# Construct the new full path
			new_file_path = os.path.join(directory, new_name + file_extension)
			
			# Rename the file
			os.rename(file_path, new_file_path)
			if(filename != new_name + file_extension):
				print(f"Renamed '{filename}' to '{new_name + file_extension}'")

def read_yolo_file(file_path, image_id):
	with open(file_path, 'r') as file:
		lines = file.readlines()
	data = []
	for line in lines:
		class_id, x_center, y_center, width, height = line.split()
		data.append([image_id, class_id, x_center, y_center, width, height])
	df = pd.DataFrame(data, columns=['image_id', 'class_id', 'x_center', 'y_center', 'width', 'height'])
	return df

def read_yolo_file_with_conf(file_path, image_id):
	with open(file_path, 'r') as file:
		lines = file.readlines()
	data = []
	for line in lines:
		class_id, x_center, y_center, width, height, conf = line.split()
		data.append([image_id, class_id, x_center, y_center, width, height, conf])
	df = pd.DataFrame(data, columns=['image_id', 'class_id', 'x_center', 'y_center', 'width', 'height', 'conf'])
	return df

def read_all_yolo_files(folder_path, conf=False):
	all_detections = []
	for file_name in os.listdir(folder_path):
		if file_name.endswith('.txt'):
			image_id = os.path.splitext(file_name)[0]
			file_path = os.path.join(folder_path, file_name)
			if conf: df_detections = read_yolo_file_with_conf(file_path, image_id)
			else:    df_detections = read_yolo_file(file_path, image_id)
			all_detections.append(df_detections)
		else:
			print(f'{file_name} is not a .txt file!')
	df_all_detections = pd.concat(all_detections, ignore_index=True)
	return df_all_detections

def convert_yolo_to_xyxy(df, dfis):
	# Converts 
	# from yolo format (xywh center normalized) 
	# to x1y1 (upper left corner) x2y2 (lower right corner) 
	df_result = pd.merge(df, dfis, on='image_id', how='inner')
	df_result['x_center_denormalized'] = df_result['x_center'].astype(float) * df_result['image_width']
	df_result['y_center_denormalized'] = df_result['y_center'].astype(float) * df_result['image_height']
	df_result['w_denormalized'] = df_result['width'].astype(float) * df_result['image_width']
	df_result['h_denormalized'] = df_result['height'].astype(float) * df_result['image_height']
	
	df_result['x1'] = df_result['x_center_denormalized'] - (df_result['w_denormalized'].astype(float)/2)
	df_result['y1'] = df_result['y_center_denormalized'] - (df_result['h_denormalized'].astype(float)/2)
	df_result['x2'] = df_result['x_center_denormalized'] + (df_result['w_denormalized'].astype(float)/2)
	df_result['y2'] = df_result['y_center_denormalized'] + (df_result['h_denormalized'].astype(float)/2)
	
	to_drop= ['class_id', 'x_center','y_center','width','height',
			  'x_center_denormalized','y_center_denormalized','w_denormalized','h_denormalized',
			  'image_width','image_height']
	df_result.drop(to_drop, axis=1, inplace=True)
	
	return df_result

def convert_yolo_to_xyxy_same_res(df, image_width, image_height):
	# Converts 
	# from yolo format (xywh center normalized) 
	# to x1y1 (upper left corner) x2y2 (lower right corner) 
	df_new = df.copy()	
	df_new['x_center_denormalized'] = df['x_center'].astype(float) * image_width
	df_new['y_center_denormalized'] = df['y_center'].astype(float) * image_height
	df_new['w_denormalized'] = df['width'].astype(float) * image_width
	df_new['h_denormalized'] = df['height'].astype(float) * image_height
	
	df_new['x1'] = df_new['x_center_denormalized'] - (df_new['w_denormalized'].astype(float)/2)
	df_new['y1'] = df_new['y_center_denormalized'] - (df_new['h_denormalized'].astype(float)/2)
	df_new['x2'] = df_new['x_center_denormalized'] + (df_new['w_denormalized'].astype(float)/2)
	df_new['y2'] = df_new['y_center_denormalized'] + (df_new['h_denormalized'].astype(float)/2)
	
	to_drop= ['class_id', 'x_center','y_center','width','height','x_center_denormalized','y_center_denormalized','w_denormalized','h_denormalized']
	df_new.drop(to_drop, axis=1, inplace=True)
	
	return df_new

def calculate_iou(box1, box2):
	"""Calculate the Intersection over Union (IoU) of two bounding boxes."""
	x_left = max(box1[0], box2[0])
	y_top = max(box1[1], box2[1])
	x_right = min(box1[2], box2[2])
	y_bottom = min(box1[3], box2[3])

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	intersection_area = (x_right - x_left) * (y_bottom - y_top)
	box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
	box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
	union_area = box1_area + box2_area - intersection_area

	return intersection_area / union_area

def evaluate_object_detection(df_model_labels_xyxy, df_gt_labels_xyxy, iou_threshold=0.5, confidence_threshold=0.5):
	# Copy dataframes to avoid modifying original data
	df_model, df_gt = df_model_labels_xyxy.copy(), df_gt_labels_xyxy.copy()

	# Filter model predictions based on the confidence threshold
	df_model_over_threshold = df_model[df_model['conf'].astype(float) >= confidence_threshold]

	# Initialize counters and storage
	tp = 0  # True Positives
	fn = 0 # False Negatives
	fp = 0  # False Positives
	model_labels_grouped = df_model_over_threshold.groupby('image_id')
	model_detected_label_indices = set()  # Indices of model detections that have been matched
	missed_persons = df_gt['image_id'].tolist()  # Track missed detections
	iou_scores = []  # Store IoU scores for later analysis
	bboxes_intersect = []  # Store matched bounding boxes for visualization
	gt_detected_labels = set()  # Store ground truth labels that have been detected

	# Loop through each ground truth label
	for gt_index, gt_row in df_gt.iterrows():
		gt_bbox = [gt_row['x1'], gt_row['y1'], gt_row['x2'], gt_row['y2']]
		
		# Check if there are predictions for the current ground truth's image
		if gt_row['image_id'] in model_labels_grouped.groups:
			model_labels_for_image = model_labels_grouped.get_group(gt_row['image_id'])
			best_iou = 0
			best_bbox_index = None

			# Loop through each predicted label in the same image
			for index, model_bbox_row in model_labels_for_image.iterrows():
				if index in model_detected_label_indices:
					continue  # Skip already matched detections
				
				model_bbox = [model_bbox_row['x1'], model_bbox_row['y1'], model_bbox_row['x2'], model_bbox_row['y2']]
				iou = calculate_iou(gt_bbox, model_bbox)

				# Check if the current IoU is the best and above the threshold
				if iou >= iou_threshold:
					if iou > best_iou:
						best_iou = iou
						best_bbox_index = index
				

			# Update matched detections and counts
			if best_bbox_index is not None:
				tp += 1
				model_detected_label_indices.add(best_bbox_index)
				gt_detected_labels.add(gt_row['image_id'])
				# Remove the image_id from missed detections
				try:
					missed_persons.remove(gt_row['image_id'])
				except ValueError:
					pass
			else:
				fn += 1
	
	# Find the false positives: where we have predicted a label that is not in the ground truth
	for index, model_bbox_row in df_model_over_threshold.iterrows():
		if index not in model_detected_label_indices:
			fp += 1

	# Calculate False Positives, False Negatives, Precision, Recall, and F1 Score
	# fp = len(df_model) - len(model_detected_label_indices)
	precision = tp / (tp + fp) if (tp + fp) > 0 else 0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0
	f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
	return tp, fp, fn, missed_persons, iou_scores, bboxes_intersect, precision, recall, f1_score

def calculate_AP_values(df_model_labels_xyxy, df_gt_labels_xyxy, fixed_conf):
	AP50_95 = 0
	precisions_IoU_50_95 = []
	recalls_IoU_50_95 = []
	for iou_threshold in np.linspace(0.5, 0.95, 10):
		iou_threshold = round(iou_threshold, 2)
		_, _, _, _, _, _, precision, recall, _  = evaluate_object_detection(df_model_labels_xyxy, df_gt_labels_xyxy, iou_threshold, fixed_conf)
		precisions_IoU_50_95.append(precision)
		recalls_IoU_50_95.append(recall)
		if(iou_threshold == 0.5):
			AP50 = precision
		elif(iou_threshold == 0.75):
			AP75 = precision
		elif(iou_threshold == 0.9):
			AP90 = precision
	AP50_95 = sum(precisions_IoU_50_95)/len(precisions_IoU_50_95)
	AR50_95 = sum(recalls_IoU_50_95)/len(recalls_IoU_50_95)
	return AP50, AP75, AP90, AP50_95, AR50_95

def run(test_path,inference_folder):
	# read ground truth labels
	gt_labels_path=test_path+'/labels'
	df_gt_labels 	= read_all_yolo_files(gt_labels_path)
	df_gt_labels_xyxy      = convert_yolo_to_xyxy_same_res(df_gt_labels, image_width=3464, image_height=2464)

	with open("res_COCO.txt", 'a') as f:
		f.write(f'\nNEW RUN\n')

	for model_labels_path in os.listdir(inference_folder):
		if model_labels_path == 'evaluated' or model_labels_path == 'evaluated_old' or model_labels_path == 'done':
			print(f'Skipping folder {model_labels_path}')
			continue
		if model_labels_path == 'exp':
			df_model_labels = read_all_yolo_files(f'{inference_folder}/{model_labels_path}/labels', conf=True)
		else:
			try:
				df_model_labels = read_all_yolo_files(f'{inference_folder}/{model_labels_path}/exp/labels', conf=True)
			except:
				df_model_labels = read_all_yolo_files(f'{inference_folder}/{model_labels_path}', conf=True)
		# Only persons
		df_model_labels = df_model_labels[df_model_labels['class_id'] == '0']

		# Normalize detections (football players detection dataset)
		df_model_labels['conf'] = df_model_labels['conf'].astype(float)
		max_conf = df_model_labels['conf'].max()
		df_model_labels['conf'] = df_model_labels['conf'] / max_conf

		df_model_labels_xyxy = convert_yolo_to_xyxy_same_res(df_model_labels, image_width=3464, image_height=2464)

		print("COCO Evaluating", model_labels_path, "in", inference_folder,  "on", test_path)
		AP50, AP75, AP90, AP50_95, AR50_95 = calculate_AP_values(df_model_labels_xyxy, df_gt_labels_xyxy, 0.99)
		print(f'{model_labels_path} & {AP50:.3f} & {AP75:.3f} & {AP50_95:.3f} & {AR50_95:.3f}')
		with open("res_COCO.txt", 'a') as f:
			f.write(f'{model_labels_path} & {AP50:.3f} & {AP75:.3f} & {AP50_95:.3f} & {AR50_95:.3f}\n')
		# move(inference_folder+model_labels_path, inference_folder+"evaluated/"+model_labels_path) # Do PASCAL AP also before moving 

def run_single(test_path,model_labels_path):
	# read ground truth labels
	gt_labels_path=test_path+'/labels'
	df_gt_labels 	= read_all_yolo_files(gt_labels_path)
	df_gt_labels_xyxy      = convert_yolo_to_xyxy_same_res(df_gt_labels, image_width=3464, image_height=2464)
	try:
		df_model_labels = read_all_yolo_files(f'{model_labels_path}', conf=True)
	except:
		df_model_labels = read_all_yolo_files(f'{model_labels_path}/exp/labels', conf=True)
	df_model_labels = df_model_labels[df_model_labels['class_id'] == '0']
	df_model_labels_xyxy = convert_yolo_to_xyxy_same_res(df_model_labels, image_width=3464, image_height=2464)

	print("COCO Evaluating", model_labels_path, "on", test_path)
	AP50, AP75, AP90, AP50_95, AR50_95 = calculate_AP_values(df_model_labels_xyxy, df_gt_labels_xyxy, 0.5)
	print(f'& {AP50_95:.3f} & {AR50_95:.3f}')
	with open("res_COCO.txt", 'a') as f:
		f.write(f'{model_labels_path} & {AP50:.3f} & {AP75:.3f} & {AP90:.3f} & {AP50_95:.3f}\n')
	# move(inference_folder+model_labels_path, inference_folder+"evaluated/"+model_labels_path) # Do PASCAL AP also before moving 


# EVALUATE ENTIRE FOLDER PRINT RESULTS
## FIMUS Inconsistent
# test_path='/home/hallvaeb/data-masteroppgave/FIMUS-Consistent-2' 
# inference_folder='/home/hallvaeb/data-masteroppgave/inferences/todo/YOLOv3-Consistent-2' 
# run_single(test_path, inference_folder)

# test_path='/home/hallvaeb/data-masteroppgave/FIMUS-Inconsistent' 
# inference_folder='/home/hallvaeb/data-masteroppgave/inferences/todo/YOLOv3-Inconsistent' 
# run_single(test_path, inference_folder)

# test_path='/home/hallvaeb/data-masteroppgave/FIMUS-Consistent-2' 
# inference_folder='/home/hallvaeb/data-masteroppgave/inferences/todo/YOLOv9-Consistent-2' 
# run_single(test_path, inference_folder)

test_path='/home/hallvaeb/data-masteroppgave/FIMUS-Consistent' 
inference_folder='/home/hallvaeb/data-masteroppgave/inferences/Consistent' 
run(test_path, inference_folder)

## COCO
# test_path='/home/hallvaeb/Code/tiling-experiment/yolov9/coco/'
# inference_folder='/home/hallvaeb/data-masteroppgave/inferences/COCO/'
# gt_labels_path=test_path+'/labels'
