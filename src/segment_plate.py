# required library
import cv2
import numpy as np
import argparse

# arg parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image file")
args = vars(ap.parse_args())
image_path = args["image"]


def preprocess_plate(image_path, blur=True, threshold=180, dilate=False):
    """
    Takes in extracted number plate from plate recognition model
    and outputs preprocessed image.
    image -> grayscale (-> blur) -> binary threshold (-> dilation)
    :param blur: set True to dilate white characters
    :param image_path: path to number plate image
    :param threshold: threshold for binary thresholding
    :param dilate: set True to dilate white characters
    :return: images of characters on number plate (in order)
    """
    # load image
    img = cv2.imread(image_path)
    # convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # scale down pixels
    img = img / 255
    # Calculates absolute values, and converts the result to 8-bit
    img = cv2.convertScaleAbs(img, alpha=(255.0))
    # grayscale image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur the image to remove noise and irrelevant information
    if blur:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    # apply inverse binary thresholding (any pixel val. smaller than
    # thresh set to 255 and vice versa)
    img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # img = cv2.adaptiveThreshold(
    #     img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2
    # )
    # dilate white region of the image (characters)
    if dilate:
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
    return img


def sort_contours(contours, reverse=False):
    """
    Grab the contour of each digit from left to right
    """
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    (contours, boundingBoxes) = zip(
        *sorted(zip(contours, boundingBoxes), key=lambda b: b[1][i], reverse=reverse)
    )
    return contours


def segment_plate(image_path, blur=True, threshold=180, dilate=False):
    """
    Takes in extracted number plate from plate recognition model
    and outputs the segmented character images from left to right.
    :param image_path: path to number plate image
    :param blur: set True to dilate white characters
    :param threshold: threshold for binary thresholding
    :param dilate: set True to dilate white characters
    :return: images of characters on number plate from left to right
    """
    # get contours of preprocessed plate (curve joining all continuous points (along the boundary)
    # sharing the same color and intensity)
    plate_img = preprocess_plate(image_path)
    contours, _ = cv2.findContours(
        plate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # create a copy version of plate_img to draw bounding box
    test_roi = cv2.imread(image_path)
    # Initialize list which will be used to append charater image
    crop_characters = []
    # define standard width and height of character
    digit_w, digit_h = 30, 60

    for c in sort_contours(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h / w
        # only select contour with defined ratio (filter irrelevent contours)
        # we know height must be greater than width of character
        if 0.75 <= ratio <= 8:
            # Select contour which has the height larger than 50% of the plate
            if h / plate_img.shape[0] >= 0.5:
                # Draw bounding box arroung digit number
                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Sperate number and give prediction
                curr_num = plate_img[y : y + h, x : x + w]
                # add black border to characters
                color = [0, 0, 0]
                # border widths
                top, bottom, left, right = [5] * 4
                curr_num = cv2.copyMakeBorder(
                    curr_num, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
                )
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                crop_characters.append(curr_num)

    print(f"Detect {len(crop_characters)} letters...")

    # cv2.imshow("boxes", test_roi)
    h = cv2.hconcat([crop_characters[i] for i in range(len(crop_characters))])
    cv2.imshow("characters", h)
    cv2.waitKey(0)


if __name__ == "__main__":
    cv2.imshow("plate", cv2.imread(image_path))
    # cv2.imshow("processed", preprocess_plate(image_path, blur=True))
    segment_plate(image_path, blur=True)
    cv2.waitKey(0)

