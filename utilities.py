import cv2
import numpy as np

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


def prepare_for_OCR(frame):
    blured1 = cv2.medianBlur(frame, 3)
    blured2 = cv2.medianBlur(frame, 51)
    divided = np.ma.divide(blured1, blured2).data
    normed = np.uint8(255 * divided / divided.max())
    th, threshed = cv2.threshold(normed, 100, 255, cv2.THRESH_OTSU)
    return threshed