
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

upper_left = (0, 550)
bottom_right = (500, 600)

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
    max_go_matches = 0
    time_start = 0
    char1 = ''
    char2 = ''

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            num_frames += 1
            # Operations on the frame start here

            # Use feature matching with ORB to get start match time
            good = compute_ORB_feature_match_for_start(frame)
            if len(good) > MIN_MATCH_COUNT and len(good) > max_go_matches:
                max_go_matches = len(good)
                time_start = num_frames / float(fps)

                frame = frame[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]


                # Once the match starts we can use the current frame to determine the characters being played
                # for filename in os.listdir('data/characters'):
                    # print(filename)
                character_template = cv2.imread('data/characters/pikachu-default.png', 1)
                character_template = cv2.cvtColor(character_template, cv2.COLOR_BGR2GRAY)
                w, h = character_template.shape[::-1]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                found = None
                # loop over the scales of the image
                for scale in np.linspace(0.2, 1.0, 20)[::-1]:
                    print(scale)
                    # resize the image according to the scale, and keep track
                    # of the ratio of the resizing
                    resized = imutils.resize(frame, width=int(frame.shape[1] * scale))
                    r = frame.shape[1] / float(resized.shape[1])

                    # if the resized image is smaller than the template, then break
                    # from the loop
                    if resized.shape[0] < h or resized.shape[1] < w:
                        break

                    # detect edges in the resized, grayscale image and apply template
                    # matching to find the template in the image
                    edged = cv2.Canny(resized, 50, 200)
                    result = cv2.matchTemplate(edged, character_template, cv2.TM_CCOEFF)
                    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

                    # check to see if the iteration should be visualized
                    # draw a bounding box around the detected region
                    clone = np.dstack([edged, edged, edged])
                    cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                                  (maxLoc[0] + w, maxLoc[1] + h), (0, 0, 255), 2)
                    cv2.imshow("Visualize", clone)
                    cv2.waitKey(0)

                    # if we have found a new maximum correlation value, then update
                    # the bookkeeping variable
                    if found is None or maxVal > found[0]:
                        found = (maxVal, maxLoc, r)

                # unpack the bookkeeping variable and compute the (x, y) coordinates
                # of the bounding box based on the resized ratio
                (_, maxLoc, r) = found
                (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
                (endX, endY) = (int((maxLoc[0] + w) * r), int((maxLoc[1] + h) * r))

                # draw a bounding box around the detected result and display the image
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.imshow("Image", frame)
                cv2.waitKey(0)



                    #cv2.imshow('frame', frame)
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


if __name__ == "__main__":
    main()
