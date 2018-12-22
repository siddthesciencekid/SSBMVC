
import cv2
import os.path
import numpy as np
import pytesseract
from download_yt import download_youtube_video

VIDEO_FILE_NAME = 'SSBM_test.mp4'

orb = cv2.ORB_create()


# GO_MATCHING
GO_TEMPLATE = cv2.imread('data/GO.jpg', 1)
kp1, des1 = orb.detectAndCompute(GO_TEMPLATE, None)
GO_FEATURE_MIN_MATCH_COUNT = 2


# ROI FRAME POSITIONS
# Ideally these frame positions would be computed from some other datapoint instead of being hardcoded
# I thought about using the size of the GO! detected at that start to generalize the region of interests
# but left that to a future task to get things working for now
STOCK_ROI_UPPER_LEFT = (0, 550)
STOCK_ROI_BOTTOM_RIGHT = (500, 600)

PLAYER1_NAME_ROI_UPPER_LEFT = (100, 0)
PLAYER1_NAME_ROI_BOTTOM_RIGHT = (225, 25)

PLAYER2_NAME_ROI_UPPER_LEFT = (735, 0)
PLAYER2_NAME_ROI_BOTTOM_RIGHT = (858, 30)

PLAYER1_STOCK_ROI_UPPER_LEFT = (25, 550)
PLAYER1_STOCK_ROI_BOTTOM_RIGHT = (195, 600)

PLAYER2_STOCK_ROI_UPPER_LEFT = (250, 550)
PLAYER2_STOCK_ROI_BOTTOM_RIGHT = (425, 600)

# Ignore template matching on the following image since these are just screen grabs
# For the template matching algorithm I was able to identify the pikachu from the downloaded stock icons
# Recognizing fox however was much trickier.

# If I just use edge detection with cv2.Canny I'm able to match fox-green but not pikachu...
# For now I have it set to use cv2.Canny with fox and pure template matching with pikachu
# Consistency would be the way to go moving forward
IGNORE_FILENAMES = ['fox-capture.PNG', 'pikachu-capture.PNG']

# Character File Name mapped to Character Variant Names
# In a full version we would have all the mappings here :)
CHARACTER_FILENAME_TO_NAME_MAP = {'fox-green.png': 'Green Fox', 'pikachu-blue.png': 'Blue Party Hat Pikachu'}

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


    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            num_frames += 1
            # Operations on the frame start here

            # As soon as the video starts we can extract the player screen names
            # using tesseract-OCR from the top of the frame
            if (num_frames == 1):

                player1_name = get_player_name(frame, PLAYER1_NAME_ROI_UPPER_LEFT, PLAYER1_NAME_ROI_BOTTOM_RIGHT)
                player2_name = get_player_name(frame, PLAYER2_NAME_ROI_UPPER_LEFT, PLAYER2_NAME_ROI_BOTTOM_RIGHT)

                print('Player 1 Screen Name: ' + player1_name)
                print('Player 2 Screen Name: ' + player2_name)

            # Use feature matching with ORB to get start match time
            if not match_has_started:
                go_feature_matches = len(compute_ORB_feature_match_for_start(frame))
            else:
                go_feature_matches = 0
            if go_feature_matches > GO_FEATURE_MIN_MATCH_COUNT:
                match_has_started = True
                time_start = num_frames / float(fps)
                char1_filename = compute_player_character(frame, PLAYER1_STOCK_ROI_UPPER_LEFT, PLAYER1_STOCK_ROI_BOTTOM_RIGHT, True)
                print('Player 1 Character: ' + CHARACTER_FILENAME_TO_NAME_MAP[char1_filename])

                char2_filename = compute_player_character(frame, PLAYER2_STOCK_ROI_UPPER_LEFT, PLAYER2_STOCK_ROI_BOTTOM_RIGHT)
                print('Player 2 Character: ' + CHARACTER_FILENAME_TO_NAME_MAP[char2_filename])


                print('game start time: ' + str(time_start) + 's')
                print('FRAME\tPLAYER 1 STOCK COUNT\tPLAYER 2 STOCK COUNT')



            # Every 10 frames analyze the frame and print statistics (stock count, in-game timer and percentage)
            if match_has_started and num_frames % 10 == 0:
                char1_stock_count, char2_stock_count = compute_num_stocks('data/characters/fox-capture.png', 'data/characters/pikachu-capture.png', frame)
                print(str(num_frames) + '\t' + str(char1_stock_count) + '\t' + str(char2_stock_count))


            cv2.imshow('frame', frame)
            if cv2.waitKey(2) & 0xFF == ord('q'):
                break
        else:
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()




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

def compute_player_character(frame, upper_left, bottom_right, use_canny = False):
    frame = frame[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]

    if use_canny:
        frame = cv2.Canny(frame, 50, 200)

    max_match = 0
    max_character = ''
    # Once the match starts we can use the current frame to determine the characters being played
    for filename in os.listdir('data/characters'):
        if not IGNORE_FILENAMES.__contains__(filename):
            character_template = cv2.imread('data/characters/' + filename, 1)
            character_template = cv2.resize(character_template, (0, 0), fx=1.41, fy=1.41)
            character_template = cv2.blur(character_template, (6, 6))

            if use_canny:
                character_template = cv2.Canny(character_template, 50, 200)



            res = cv2.matchTemplate(frame, character_template, cv2.TM_CCOEFF_NORMED)

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > max_match:
                max_match = max_val
                max_character = filename
    return max_character

def get_player_name(frame, upper_left, bottom_right):
    config = ('-l eng --oem 1 --psm 3')
    frame = frame[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]

    # Make grayscale and invert (black text on white bg) image so that tesseract-OPR can
    # perform better
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.bitwise_not(frame)

    cv2.imshow('frame', frame)
    cv2.waitKey(0)


    text = pytesseract.image_to_string(frame, config=config)

    return text


if __name__ == "__main__":
    main()
