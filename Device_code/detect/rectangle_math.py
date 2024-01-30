from collections import namedtuple

Rectangle = namedtuple("Rectangle", "xmin ymin xmax ymax")

# used to decide how close two boxes in adjacent splits have to be 
# in order to be considered for merging
OVERLAP_THRESHOLD = 75
# detections that are the size of subimages/tiles should have a higher
# confidence threshold to be accpeted ias a person, since there are some
# issues with yolo making large detections that are wrong (large as in the size of a tile)
CONF_THRESHOLD_FOR_LARGE_BOXES = 0.85

# ra = Rectangle(0., 0., 2., 2.)
# rb = Rectangle(1., 0., 3., 2.)
# # intersection here is (3, 3, 4, 3.5), or an area of 1*.5=.5


def create_rect_from_box(box):
    return Rectangle(box[0], box[1], box[2], box[3])


# returns intersection percentage of two rectangles (a,b)
def calc_intersect_percentage(a, b):  # returns None if rectangles don't intersect

    area = get_area(a)

    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx >= 0) and (dy >= 0):
        areaIntersect = dx * dy
        return areaIntersect / area * 100.0


# returns area of rectangle (a)
def get_area(a):
    lineX = a.xmax - a.xmin
    lineY = a.ymax - a.ymin
    return lineX * lineY


def remove_boxes_inside_main(main_detections, detections_tiles):
    print("Removing duplicate bounding boxes..")
    for i in main_detections:
        j = 0
        while j < len(detections_tiles):
            percentage = calc_intersect_percentage(
                create_rect_from_box(detections_tiles[j].box), create_rect_from_box(i.box)
            )
            if percentage is not None and percentage > 70:
                del detections_tiles[j]
                j -= 1
            j += 1


def remove_boxes_inside_other_boxes(detections):
    i = 0
    while i < len(detections):
        j = 0
        while j < len(detections):
            if i != j:
                percent = calc_intersect_percentage(
                    create_rect_from_box(detections[j].box), create_rect_from_box(detections[i].box)
                )
                if percent is not None and percent > 70:
                    area1 = get_area(create_rect_from_box(detections[i].box))
                    area2 = get_area(create_rect_from_box(detections[j].box))
                    if area1 >= area2:
                        del detections[j]
                        if j < i:
                            i -= 1
                        j -= 1
            j += 1
        i += 1


# length height of full image
def remove_big_boxes(detections, length, height):
    image_area = length * height
    threshold = int(image_area / 4)
    i = 0
    while i < len(detections):
        area = get_area(create_rect_from_box(detections[i].box))
        if area > threshold:
            del detections[i]
            i -= 1
        i += 1

# Sometimes yolo detects boxes that are about the size of an image tile.
# These are likely not humans, so they should be deleted.
# param: detections is 3 dimensional array of Detection objects
# param: tiled_image is 3 dimensional array of image tiles (numpy arrays) 
# this is used for determining the size of the tile/image
# param: remove_threshhold is how big the detection has to be relative to image/tile
# size, in order to be removed
def check_for_splitsize_boxes(detections, tiled_image, remove_threshhold):
    removed_detections = []
    for i in range(len(detections)):
        for j in range(len(detections[i])):
            # tile i j. check if any detections here are too big for it to be likely
            # that it is a human
            max_length = tiled_image[j][i].shape[1]
            max_height = tiled_image[j][i].shape[0]
             
            k = 0
            while k < len(detections[i][j]):
                detection = detections[i][j][k]
                detection_length = detection.box[2] - detection.box[0]
                detection_height = detection.box[3] - detection.box[1]

                percent_length = detection_length / max_length * 100
                percent_height = detection_height / max_height * 100

                if(percent_length > (100 - remove_threshhold) and percent_height > (100 - remove_threshhold)):
                    if(detection.conf < CONF_THRESHOLD_FOR_LARGE_BOXES):
                        # confidence wasnt high enough
                        removed_detections.append(detection)
                        del detections[i]
                        k -= 1
                k += 1
    return removed_detections 

# Sometimes yolo detects boxes that are about the size of an image tile.
# These are likely not humans, so they should be deleted.
# param: detections is 3 dimensional array of Detection objects
# param: tiled_image is 3 dimensional array of image tiles (numpy arrays) 
# this is used for determining the size of the tile/image
# param: remove_threshhold is how big the detection has to be relative to image/tile
# size, in order to be removed
def check_for_splitsize_boxes_in_flat_list(detections, tiled_image, remove_threshhold):
    removed_detections = []
    max_length = tiled_image[0][0].shape[1]
    max_height = tiled_image[0][0].shape[0]
    
    i = 0
    while i < len(detections):
        detection = detections[i]
        detection_length = detection.box[2] - detection.box[0]
        detection_height = detection.box[3] - detection.box[1]

        percent_length = detection_length / max_length * 100
        percent_height = detection_height / max_height * 100

        if(percent_length > (100 - remove_threshhold) and percent_height > (100 - remove_threshhold) and percent_length < (100 + remove_threshhold) and percent_length < (100 + remove_threshhold)):
            if(detection.conf < CONF_THRESHOLD_FOR_LARGE_BOXES):
                # confidence wasnt high enough
                removed_detections.append(detection)
                del detections[i]
                i -= 1
        i+=1
    return removed_detections 


# pixels is the limit of pixels between two boxes, that are
# believed to describe a single person
def merge_boxes_belonging_to_the_same_person(
    boxes, width, height, horizontal_amt, vertical_amt, pixels
):
    """
    Does merging on boxes that should be merged in the horizontal direction
    Then joins boxes in horizontal tiles, removing the vertical walls in the grid
    Does merging on boxes in the remaining tiles (only rows left)
    :param boxes: boxes: 3d array containing coordinates for the detections.
    1st dimension is horizontal tiles. 2nd vertical tiles.
    3rd is detections in specific tile
    :param confs: confs: 3d array of corresponding confidences (to boxes)
    :param clss: 3d array of corresponding classes (to boxes)
    :param width: width of full original image
    :param height: height of full original image
    :param horizontal_amt: amount of tiles each rows (horizontal)
    :param vertical_amt: amount of tiles each column (horizontal)
    :param pixels: threshold for how close a box must be to a split-border to be considered
    for merging. Smaller values means boxes have to be closer to the edge to be considered
    :return: new_boxes: 2d array containing the boxes including the merged boxes
    new_confs, new_clss: corresponding 2d arrays containing confidences, and yolo-classes of detections
    """
    vertical_merging(boxes, width, horizontal_amt, pixels)
    new_boxes = remove_vertical_walls(
        boxes, horizontal_amt, vertical_amt
    )
    horizontal_merging(new_boxes, height, vertical_amt, pixels)

    return flatten_2d_array(new_boxes)


def vertical_merging(detections, width, horizontal_amt, pixels):
    """
    iterates 3d array (boxes) and checks if the boxes in each tile should
    merge with other boxes in the tile on its right
    :param detections: 3d array containing Detection objects containing
    coordinates, confidences and yolo classes of the detections.
    1st dimension is horizontal tiles. 2nd vertical tiles.
    3rd is detections in specific tile
    :param width: width of image
    :param horizontal_amt: amount of tiles in the rows (horizontal)
    :param pixels: threshold for how close a box must be to a split-border to be considered
    for merging. Smaller values means boxes have to be closer to the edge to be considered
    """
    length_slice_x = int(width / horizontal_amt)
    # subtract 1 since we dont want to check image splices to the right of the right-most splices
    for i in range(len(detections) - 1):
        for j in range(len(detections[i])):
            vertical_border_x_val = length_slice_x * (i + 1)
            k = 0
            # for k in range(len(boxes[i][j])):
            while k < len(detections[i][j]):
                # check for any boxes near the right border of the current image splice
                if detections[i][j][k].box[2] > vertical_border_x_val - pixels:
                    # check for related boxes on the image splice on the right
                    # rect[1] and [3] are the y values of the two coordinates sets of the rectangle
                    indices = check_for_related_box(
                        detections[i][j][k].box[1],
                        detections[i][j][k].box[3],
                        detections[i + 1][j],
                        vertical_border_x_val,
                        pixels,
                    )
                    if len(indices) > 0:
                        merge_boxes(detections, [i, j, k], indices)
                        del detections[i][j][k]
                        k -= 1
                k += 1

# almost the same function as vertical_merging. only one less dimension in detections array
# and therefore one less nested loop
def horizontal_merging(detections, height, vertical_amt, pixels):
    i = 0
    height_slice_y = int(height / vertical_amt)

    for j in range(vertical_amt - 1):
        horizontal_border_y_val = height_slice_y * (j + 1)
        k = 0
        while k < len(detections[j]):
            # check for any boxes near the bottom border of the current image splice
            if detections[j][k].box[3] > horizontal_border_y_val - pixels:
                # check for related boxes on the image splice below
                # rect[0] and [2] are x-min and x-max of the rectangle
                indices = check_for_related_box_horizontal_line(
                    detections[j][k].box[0],
                    detections[j][k].box[2],
                    detections[j + 1],
                    horizontal_border_y_val,
                    pixels,
                )
                if len(indices) > 0:
                    merge_boxes_horizontal_splits(
                        detections, [i, j, k], indices
                    )
                    del detections[j][k]
                    k -= 1
            k += 1

def remove_vertical_walls(detections, horizontal_amt, vertical_amt):
    """
    Joins horizontal tiles, making the 3d array into a 2d array.
    :param detections: 3d array containing coordinates for the detections
    :param confs: 3d array of corresponding confidences
    :param clss: 3d array of corresponding yolo classes (0=human)
    :param horizontal_amt: amount of tiles in a row (horizontal)
    :param vertical_amt: amount of tiles in a column (vertical)
    :return: boxes, confs and clss where horizontal tiles are joined,
    making them 2d arrays, instead of 3d arrays
    """
    detections_2d = []
    i = 0
    for j in range(
        vertical_amt
    ):  # vertical iteration (reversed compared to earlier where horizontal was iterated first)
        horizontal_splits_detections = []
        for i in range(horizontal_amt):  # horizontal iteration
            for k in range(len(detections[i][j])):
                horizontal_splits_detections.append(detections[i][j][k])

        detections_2d.append(horizontal_splits_detections)
    return detections_2d

def check_for_related_box(y1, y2, detections_in_tile, vertical_border_x_val, pixels):
    """
    checks for suitable boxes for merging
    :param y1: y-min of the box that is evaluated against boxes (boxes_in_tile)
    :param y2: y-max of the box that is evaluated against boxes (boxes_in_tile)
    :param boxes_in_tile: list of boxes to check if they should be merged with
    the box being evaluated
    :param vertical_border_x_val: x value of the current line that seperates boxes.
    vertically. If the boxes are close enough to this line they can merge
    :param pixels: threshold for closeness to the vertical line the boxes have to be
    to evaluated for merging
    :return: indices of boxes in this tile to be merged with the box being evaluated
    """
    indices_of_related_boxes = []
    for i in range(len(detections_in_tile)):
        if detections_in_tile[i].box[0] < vertical_border_x_val + pixels:
            overlap = calculate_overlap_percentage_of_lines(
                y1, y2, detections_in_tile[i].box[1], detections_in_tile[i].box[3]
            )
            overlap2 = calculate_overlap_percentage_of_lines(
                detections_in_tile[i].box[1], detections_in_tile[i].box[3], y1, y2
            )
            if overlap > OVERLAP_THRESHOLD or overlap2 > OVERLAP_THRESHOLD:
                # this box should be merged. add its index to list
                indices_of_related_boxes.append(i)
    return indices_of_related_boxes


def check_for_related_box_horizontal_line(
    y1, y2, detections_in_tile, horizontal_border_y_val, pixels
):
    indices_of_related_boxes = []
    for i in range(len(detections_in_tile)):
        if detections_in_tile[i].box[1] < horizontal_border_y_val + pixels:
            overlap = calculate_overlap_percentage_of_lines(
                y1, y2, detections_in_tile[i].box[0], detections_in_tile[i].box[2]
            )
            overlap2 = calculate_overlap_percentage_of_lines(
                detections_in_tile[i].box[0], detections_in_tile[i].box[2], y1, y2
            )
            if overlap > OVERLAP_THRESHOLD or overlap2 > OVERLAP_THRESHOLD:
                # this box should be merged. add its index to list
                indices_of_related_boxes.append(i)
    return indices_of_related_boxes


def calculate_overlap_percentage_of_lines(min1, max1, min2, max2):
    """
    calculates the percentage of overlap between two lines.
    line 1 (min1,max1)
    line 2 (min2,max2)
    """
    overlap = max(0, min(max1, max2) - max(min1, min2))
    length = max1 - min1
    return overlap / length * 100


# val_box i,j,k
def merge_boxes(detections, val_box, indices_of_other_boxes):
    """
    merges the box retrieved from (all_boxes) using the values of (val_box)
    with the boxes found in (all_boxes) using the values from (val_box) and
    (indices for other boxes). The leftover boxes used in the merging are deleted
    from the 3d arrays (all_boxes, confs, clss)
    :param all_boxes: 3d array containing coordinates of the detections/boxes
    :param confs: 3d array of corresponding confidences
    :param clss: 3d array of corresponding classes
    :param val_box: Structure: [x,y,b]. indices used to find the box in (all_boxes) and
    the boxes in the adjacent tile
    :param indices_of_other_boxes:
    """
    # the box being transformed/enlarged (the resulting box from the merge) belongs to the tile to
    # the right of tile with the main box. this box is chosen as the one being transformed
    # since when it is transformed to the bigger merged box it will be located
    # in a tile that hasn't been checked yet, enabling merging of already merged boxes across multiple
    # tiles
    main_box = detections[val_box[0] + 1][val_box[1]][indices_of_other_boxes[0]].box
    box_from_current_splice = detections[val_box[0]][val_box[1]][val_box[2]].box
    do_merge(main_box, box_from_current_splice)

    for i in range(len(indices_of_other_boxes)):
        if i != 0:
            # other_box structure: [xMin, yMin, xMax, yMax]
            other_box = detections[val_box[0] + 1][val_box[1]][indices_of_other_boxes[i]].box
            do_merge(main_box, other_box)

            del detections[val_box[0] + 1][val_box[1]][indices_of_other_boxes[i]]

# val_box i,j,k
def merge_boxes_horizontal_splits(
    detections, val_box, indices_of_other_boxes
):
    # the first box to be merged belonging to the splice to the right of splice being checked.
    # this box is chosen as the main since when transformed to the bigger merged box it will be located
    # in a splice that hasnt been checked yet, enabling merging of boxes across multiple splices
    main_box = detections[val_box[1] + 1][indices_of_other_boxes[0]].box
    box_from_current_splice = detections[val_box[1]][val_box[2]].box
    do_merge(main_box, box_from_current_splice)
    for i in range(len(indices_of_other_boxes)):
        if i != 0:
            # actual box [xMin, yMin, xMax, yMax]
            other_box = detections[val_box[1] + 1][indices_of_other_boxes[i]].box
            do_merge(main_box, other_box)
            print("\n\nATTENTION\n\n")
            del detections[val_box[1] + 1][indices_of_other_boxes[i]]


def do_merge(box1, box2):
    """
    calculates the x-min, y-min, x-max, y-max of two rectangles/boxes
    :param box1: structure [x-min, y-min, x-max, y-max] this is the box that is transformed
    :param box2: structure [x-min, y-min, x-max, y-max]
    """
    if box2[0] < box1[0]:
        box1[0] = box2[0]
    if box2[1] < box1[1]:
        box1[1] = box2[1]
    if box2[2] > box1[2]:
        box1[2] = box2[2]
    if box2[3] > box1[3]:
        box1[3] = box2[3]


def flatten_2d_array(array):
    """
	Flattens two-dimensional array to a one-dimensional array. Preprocess for drawing boxes.
	@param array:
	@return: flatten array
	"""
    result = []
    for i in array:
        for j in i:
            result.append(j)
    return result


"""
boxes = [[[0, 0, 9, 20]]], [[[11, 0, 19, 10], [11, 11, 19, 20]]], [[[21, 0, 29, 10]]]
confs = [[1]], [[1, 1]], [[1]]
clss = [[1]], [[1, 1]], [[1]]"""

# boxes = [[[], [[82, 948, 334, 1366], [518, 990, 606, 1332]], [[88, 1368, 260, 1507]], []], [[], [[609, 948, 708, 1366], [945, 907, 1214, 1366]], [[847, 1368, 1214, 2050], [608, 1368, 683, 1480]], []], [[], [], [], []], [[], [[1951, 928, 2088, 1259], [2269, 879, 2430, 1363]], [], []], [[], [[2435, 866, 2585, 1347]], [], []], [[], [[3378, 893, 3491, 1239]], [], []]]

# confs =  [[[], [0.93626046, 0.92951846], [0.74845272], []], [[], [0.98199797, 0.86902577], [0.94229621, 0.88828266], []], [[], [], [], []], [[], [0.98532993, 0.94907194], [], []], [[], [0.86173189], [], []], [[], [0.98449796], [], []]]

# clss =  [[[], [0.93626046, 0.92951846], [0.74845272], []], [[], [0.98199797, 0.86902577], [0.94229621, 0.88828266], []], [[], [], [], []], [[], [0.98532993, 0.94907194], [], []], [[], [0.86173189], [], []], [[], [0.98449796], [], []]]

# merge_boxes_belonging_to_the_same_person(boxes, confs, clss, 3648, 2736, 6, 4, 4)

""" scenario below from football5.jpg 6x4 4 pixel threshhold"""
# boxes = [[[], [[574, 778, 720, 998], [194, 805, 346, 998]], [[534, 1000, 696, 1074], [189, 1000, 332, 1166]]],
#         [[], [[1389, 833, 1446, 980], [1448, 905, 1498, 998]], []]]
# confs = [[[], [0.98827785, 0.92151892], [0.65193284, 0.55609834]], [[], [0.92027873, 0.3914499], []]]
# clss = [[[], [0.98827785, 0.92151892], [0.65193284, 0.55609834]], [[], [0.92027873, 0.3914499], []]]
# merge_boxes_belonging_to_the_same_person(boxes, confs, clss, 1500, 1500, 2, 3, 4)
