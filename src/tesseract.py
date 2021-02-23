# required library
import cv2
import numpy as np
import argparse
from segment_plate import preprocess_plate
import pytesseract


# arg parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image file")
args = vars(ap.parse_args())
image_path = args["image"]

if __name__ == "__main__":
    cv2.imshow("plate", cv2.imread(image_path))
    cv2.imshow("processed", preprocess_plate(image_path))

    img = cv2.imread(image_path)
    img = preprocess_plate(image_path)
    text = pytesseract.image_to_string(img, config="--psm 11")
    print("License Plate Recognition\n")
    print("Detected license plate Number is:", text)

