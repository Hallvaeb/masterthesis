def convert_coordinates(
    original_x_coordinate,
    original_y_coordinate,
    overlapPercentage,
    tileX,
    tileY,
    height,
    width,
    sliceAmtX,
    sliceAmtY,
):
    """
    Converts coordinates from a tile to the corresponding coordinates on the full image
        @param original_x_coordinate:
        @param original_y_coordinate:
        @param overlapPercentage:
        @param tileX:
        @param tileY:
        @param height:
        @param width:
        @param sliceAmtX:
        @param sliceAmtY:
        @return: coordinates
    """
    print("Converting coordinates..")
    new_x = compute_coordinate(
        original_x_coordinate, width, sliceAmtX, tileX, overlapPercentage
    )
    new_y = compute_coordinate(
        original_y_coordinate, height, sliceAmtY, tileY, overlapPercentage
    )
    return new_x, new_y


def compute_coordinate(
    coordinatePosition, imageSize, sliceAmount, imagePosition, overlap
):
    """
    Helper function for converting coordinates
    @param coordinatePosition:
    @param imageSize:
    @param sliceAmount:
    @param imagePosition:
    @param overlap:
    @return: coordinate
    """
    length_of_slice = int(imageSize / sliceAmount)

    x_add = imagePosition * length_of_slice
    if imagePosition != 0:
        x_add -= int(length_of_slice * overlap)
    return coordinatePosition + x_add


# Images are being shrunk to use less space. This function is used
# to shrink the detections so they match the shrunk image
def compute_coordinates_of_shrunk_image(detections, shrink_amount):
    for detection in detections:
        for j in range(len(detection.box)):
            detection.box[j] = int(round(detection.box[j]/shrink_amount))

