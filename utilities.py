# Helpful utility functions

import cv2
import numpy as np
import pytesseract

# Catches and fixes some common errors that the OCR
# gives when attempting to read the time
def fix_time(time):
    if len(time) >= 1:
        time = time.replace('a', '3')
        time = time.replace('T', '7')
        time = time.replace('O', '0')


        if time[1] != ':':
            time = time[:1] + ':' + time[1:]

        if not time.__contains__(' '):
            time = time[:4] + ' ' + time[4:]
        return time[:7]
    else:
        return 'N/A'

# Prepares an image for OCR
# by blurring with different kernels and dividing
# and thresholding as described in this stack overflow answer
# https://stackoverflow.com/questions/52459794/proper-image-thresholding-to-prepare-it-for-ocr-in-python-using-opencv
def prepare_for_OCR(frame):
    blured1 = cv2.medianBlur(frame, 3)
    blured2 = cv2.medianBlur(frame, 51)
    divided = np.ma.divide(blured1, blured2).data
    normed = np.uint8(255 * divided / divided.max())
    th, threshed = cv2.threshold(normed, 100, 255, cv2.THRESH_OTSU)
    return threshed

# Returns the percentage formatted version of the percentage if it is found
# or 'N/A' if OCR is unable to determine the percentage
def get_percentage(percentage):
    if 0 < len(percentage) < 4:
        return percentage + '%'
    elif len(percentage) > 4:
        return 'N/A'
    else:
        return '0%'

# Uses tesseract OCR to grab the text from an image
# If normalize_bg is passed in as True, the prepare_for_OCR algorithm
# will be used
def get_text_from_image(frame, upper_left, bottom_right, normalize_bg = False):
    config = ('-l eng --oem 1 --psm 8')
    frame = frame[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]

    # Make grayscale and invert (black text on white bg) image so that tesseract-OPR can
    # perform better
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.bitwise_not(frame)

    if normalize_bg:
        frame = cv2.multiply(frame, 1.5)
        frame = prepare_for_OCR(frame)


    text = pytesseract.image_to_string(frame, config=config)
    return text

