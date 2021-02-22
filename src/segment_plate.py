# required library
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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
        if 0.75 <= ratio <= 3.5:
            # Select contour which has the height larger than 50% of the plate
            if h / plate_img.shape[0] >= 0.5:
                # Draw bounding box arroung digit number
                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Sperate number and give prediction
                curr_num = plate_img[y : y + h, x : x + w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(
                    curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                crop_characters.append(curr_num)

    print(f"Detect {len(crop_characters)} letters...")
    cv2.imshow("processed", test_roi)
    cv2.waitKey(0)

    fig = plt.figure(figsize=(14, 4))
    grid = gridspec.GridSpec(ncols=len(crop_characters), nrows=1, figure=fig)

    for i in range(len(crop_characters)):
        fig.add_subplot(grid[i])
        plt.axis(False)
        plt.imshow(crop_characters[i], cmap="gray")
    plt.show()


if __name__ == "__main__":
    test_image_path = "../input/num1.png"
    cv2.imshow("plate", cv2.imread(test_image_path))
    cv2.imshow("processed", preprocess_plate(test_image_path))
    cv2.waitKey(0)
    segment_plate(test_image_path, blur=False)
