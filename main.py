
import cv2
import os.path
import numpy as np
import imutils
from matplotlib import pyplot as plt
from download_yt import download_youtube_video

VIDEO_FILE_NAME = 'SSBM_test.mp4'
GO_TEMPLATE = cv2.imread('data/GO.jpg', 1)
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(GO_TEMPLATE, None)
MIN_MATCH_COUNT = 2

# ROI FRAME POSITIONS
STOCK_ROI_UPPER_LEFT = (0, 550)
STOCK_ROI_BOTTOM_RIGHT = (500, 600)

PLAYER1_NAME_ROI_UPPER_LEFT = (100, 0)
PLAYER1_NAME_ROI_BOTTOM_RIGHT = (225, 40)

def main():

    # Determine if the test clip is already downloaded and download the clip if it isn't
    if not os.path.isfile('data/' + VIDEO_FILE_NAME):
        # Youtube clip that we're analyzing https://www.youtube.com/watch?v=bj7IX18ccdY
        download_youtube_video('bj7IX18ccdY', VIDEO_FILE_NAME)

    # Use VideoCapture to stream/read the video
    cap = cv2.VideoCapture('data/' + VIDEO_FILE_NAME)
    # Get the FPS of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Datapoints
    num_frames = 0
    match_has_started = False
    max_go_matches = 0
    time_start = 0
    char1 = ''
    char2 = ''
    char1_stock_count = 0
    char2_stock_count = 0

    print('FRAME\tPLAYER 1 STOCK COUNT\tPLAYER 2 STOCK COUNT')
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            num_frames += 1
            # Operations on the frame start here

            # Use feature matching with ORB to get start match time
            good = compute_ORB_feature_match_for_start(frame)
            if len(good) > MIN_MATCH_COUNT and len(good) > max_go_matches:
                match_has_started = True
                max_go_matches = len(good)
                time_start = num_frames / float(fps)

                frame = frame[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]
                cv2.imshow('frame', frame)
                cv2.waitKey(0)



                # Once the match starts we can use the current frame to determine the characters being played
                # for filename in os.listdir('data/characters'):
                    # print(filename)
                # character_template = cv2.imread('data/characters/pikachu-capture.png', 1)
                # character_template = cv2.cvtColor(character_template, cv2.COLOR_BGR2GRAY)
                # w, h = character_template.shape[::-1]
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #
                # res = cv2.matchTemplate(frame, character_template, cv2.TM_CCOEFF_NORMED)
                #
                # # Specify a threshold
                # threshold = 0.7
                #
                # # Store the coordinates of matched area in a numpy array
                # loc = np.where(res >= threshold)
                #
                # found = set()
                # # Draw a rectangle around the matched region.
                # for pt in zip(*loc[::-1]):
                #     cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
                #     sensitivity = 43
                #     found.add((round(pt[0] / sensitivity), round(pt[1] / sensitivity)))



                #cv2.imshow('Detected', frame)
                #cv2.waitKey(0)

            # Every 10 frames analyze the frame and print statistics (stock count, in-game timer and percentage)
            if match_has_started and num_frames % 10 == 0:
                char1_stock_count, char2_stock_count = compute_num_stocks('data/characters/fox-capture.png', 'data/characters/pikachu-capture.png', frame)
                print(str(num_frames)  + 's\t' + str(char1_stock_count) + '\t' + str(char2_stock_count))


            cv2.imshow('frame', frame)
            if cv2.waitKey(2) & 0xFF == ord('q'):
                break
        else:
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    print('game start time: ' + str(time_start) + 's')




def compute_ORB_feature_match_for_start(frame):
    # find the keypoints and descriptors with ORB
    kp2, des2 = orb.detectAndCompute(frame, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Apply ratio test here to find key matches
    good = []
    for m in matches:
        if m.distance < 11:
            good.append([m])
    return good

def compute_num_stocks(player1_stock_image, player2_stock_image, frame):
    frame = frame[STOCK_ROI_UPPER_LEFT [1]: STOCK_ROI_BOTTOM_RIGHT[1], STOCK_ROI_UPPER_LEFT [0]: STOCK_ROI_BOTTOM_RIGHT[0]]

    character_template1 = cv2.imread(player1_stock_image, 1)
    character_template1 = cv2.cvtColor(character_template1, cv2.COLOR_BGR2GRAY)
    w1, h1 = character_template1.shape[::-1]
    character_template2 = cv2.imread(player2_stock_image, 1)
    character_template2 = cv2.cvtColor(character_template2, cv2.COLOR_BGR2GRAY)
    w2, h2 = character_template2.shape[::-1]

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    res1 = cv2.matchTemplate(frame, character_template1, cv2.TM_CCOEFF_NORMED)
    res2 = cv2.matchTemplate(frame, character_template2, cv2.TM_CCOEFF_NORMED)

    # Specify a threshold for which the template should match the frame
    threshold = 0.80

    # Store the coordinates of matched area in a numpy array
    loc1 = np.where(res1 >= threshold)
    loc2 = np.where(res2 >= threshold)

    player1_found = set()
    player2_found = set()
    sensitivity = (w2 * 2) - 1

    for pt in zip(*loc1[::-1]):
        player1_found.add((round(pt[0] / sensitivity), round(pt[1] / sensitivity)))

    for pt in zip(*loc2[::-1]):
        player2_found.add((round(pt[0] / sensitivity), round(pt[1] / sensitivity)))

    return len(player1_found), len(player2_found)



if __name__ == "__main__":
    main()
