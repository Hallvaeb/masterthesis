import numpy as np
import cv2


def image_slicer(img):
    list_of_images = []
    tiles = image_slicer.slice(img, 6, save=False)

    for tile in tiles:
        list_of_images.append(tile)

    return list_of_images


def image_stitcher(listOfImages):
    img_combined = image_slicer.join(listOfImages)
    return img_combined.convert("RGB")


def upscale(img):
    print("Original Dimensions : ", img.shape)
    scale_percent = 200  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    print("Resized Dimensions : ", resized.shape)
    return resized


def gamma_correction(img):
    print("Applying gamma correction..")
    look_up_table = np.empty((1, 256), np.uint8)
    for i in range(256):
        look_up_table[0, i] = np.clip(pow(i / 255.0, 1.2) * 255.0, 10, 255)
    return cv2.LUT(img, look_up_table)


def draw_people_count(img, amount_of_people):
    """Draws peoplecount center screen"""
    font = cv2.FONT_HERSHEY_PLAIN
    line = cv2.LINE_AA
    ppl_text = "People(s): {:d}".format(amount_of_people)
    cv2.putText(img, ppl_text, (281, 20), font, 1.0, (32, 32, 32), 4, line)
    cv2.putText(img, ppl_text, (280, 20), font, 1.0, (240, 240, 240), 1, line)
    return img

def draw_text_on_image(img, text):
    """Writes text on the image"""
    font = cv2.FONT_HERSHEY_PLAIN
    line = cv2.LINE_AA
    cv2.putText(img, text, (5, img.shape[0]-10), font, 1.0, (32, 32, 32), 4, line)
    cv2.putText(img, text, (5, img.shape[0]-10), font, 1.0, (240, 240, 240), 1, line)
    return img

# split image
def split_image(test_image, splicesX, splicesY, overlap):
    print("Tiling images..")
    # image = Image.open("testimage1920.jpg")
    # buffered = BytesIO()
    # image.save(buffered, format="JPEG")
    # imgB64 = base64.b64encode(buffered.getvalue())
    # test_image = numpy.array(image)
    x = splicesX
    y = splicesY

    # Image chunk dimensions
    windowsize_c = int(float(test_image.shape[1]) / x - 1)
    windowsize_r = int(float(test_image.shape[0]) / y - 1)

    pix_x = int(windowsize_c * overlap)
    pix_y = int(windowsize_r * overlap)
    # print("pix", windowsize_r, pix_y)

    images = []
    # print("images", images)

    i = 0
    for r in range(0, test_image.shape[0] - windowsize_r, windowsize_r):
        image_list = []
        for c in range(0, test_image.shape[1] - windowsize_c, windowsize_c):
            window = None
            # print(r)
            # print("max", test_image.shape[0] - windowsize_r - y)
            x_subtract = pix_x
            x_add = pix_x
            y_subtract = pix_y
            y_add = pix_y
            if c == 0:
                x_subtract = 0
            elif c == test_image.shape[1] - windowsize_c - x:
                x_add = 0

            if r == 0:
                y_subtract = 0
            elif r == test_image.shape[0] - windowsize_r - y:
                y_add = 0

            window = test_image[
                r - y_subtract : r + windowsize_r + y_add,
                c - x_subtract : c + windowsize_c + x_add,
            ]

            image_list.append(window)
            i += 1
        images.append(image_list)
    return images
