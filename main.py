import cv2
import os.path
import numpy as np
import pytesseract

# Import helper functions
from utilities import fix_time, get_text_from_image, get_percentage, prepare_for_OCR
from download_yt import download_youtube_video


VIDEO_FILE_NAME = 'SSBM_test.mp4'

# For feature matching we will use ORB
ORB = cv2.ORB_create()


# To figure out the match start time we will use
# feature matching with this template
GO_TEMPLATE = cv2.imread('data/GO.jpg', 1)
kp1, des1 = ORB.detectAndCompute(GO_TEMPLATE, None)
GO_FEATURE_MIN_MATCH_COUNT = 2


# ROI FRAME POSITIONS
# Ideally these frame positions would be computed from some other datapoint instead of being hardcoded
# I thought about using the size of the GO! detected at that start to generalize the region of interests
# but left that to a future task to get things working for now
STOCK_ROI_UPPER_LEFT = (0, 550)
STOCK_ROI_BOTTOM_RIGHT = (500, 600)

PLAYER1_NAME_ROI_UPPER_LEFT = (100, 0)
PLAYER1_NAME_ROI_BOTTOM_RIGHT = (225, 27)

PLAYER2_NAME_ROI_UPPER_LEFT = (744, 0)
PLAYER2_NAME_ROI_BOTTOM_RIGHT = (854, 23)

PLAYER1_STOCK_ROI_UPPER_LEFT = (25, 550)
PLAYER1_STOCK_ROI_BOTTOM_RIGHT = (195, 600)

PLAYER2_STOCK_ROI_UPPER_LEFT = (250, 550)
PLAYER2_STOCK_ROI_BOTTOM_RIGHT = (425, 600)

GAME_TIMER_ROI_UPPER_LEFT = (400, 55)
GAME_TIMER_ROI_BOTTOM_RIGHT = (625, 110)

PLAYER1_PERCENTAGE_ROI_UPPER_LEFT = (95, 620)
PLAYER1_PERCENTAGE_ROI_BOTTOM_RIGHT = (200, 690)

PLAYER2_PERCENTAGE_ROI_UPPER_LEFT = (320, 620)
PLAYER2_PERCENTAGE_ROI_BOTTOM_RIGHT = (420, 690)

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

# Setting this variable to True will cause the program to display the video
# in a separate frame as the analysis is being performed
PLAY_VIDEO = True

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
                player1_name = get_text_from_image(frame, PLAYER1_NAME_ROI_UPPER_LEFT, PLAYER1_NAME_ROI_BOTTOM_RIGHT)
                player2_name = get_text_from_image(frame, PLAYER2_NAME_ROI_UPPER_LEFT, PLAYER2_NAME_ROI_BOTTOM_RIGHT)

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
                print('FRAME\tGAME TIMER\tPLAYER 1 STOCK COUNT\tPLAYER 2 STOCK COUNT\tPLAYER 1 PERCENTAGE\tPLAYER 2 PERCENTAGE')



            # Every 10 frames analyze the frame and print statistics (stock count, in-game timer and percentage)
            # Match ends at around frame 1930 so we stop printing stats at that time
            if match_has_started and num_frames % 10 == 0 and num_frames <= 1930:
                # Get the time from the timer ROI and fix as needed
                time = get_text_from_image(frame, GAME_TIMER_ROI_UPPER_LEFT, GAME_TIMER_ROI_BOTTOM_RIGHT, True)
                time = fix_time(time)

                # Get the damage percentages from the respective ROIs and fix as needed
                player1_percentage = compute_dmg_percentages(frame, PLAYER1_PERCENTAGE_ROI_UPPER_LEFT,
                                                             PLAYER1_PERCENTAGE_ROI_BOTTOM_RIGHT)
                player2_percentage = compute_dmg_percentages(frame, PLAYER2_PERCENTAGE_ROI_UPPER_LEFT,
                                                             PLAYER2_PERCENTAGE_ROI_BOTTOM_RIGHT)

                player1_percentage = get_percentage(player1_percentage)
                player2_percentage = get_percentage(player2_percentage)

                # Compute the number of stocks each player has
                char1_stock_count, char2_stock_count = compute_num_stocks('data/characters/fox-capture.png',
                                                                          'data/characters/pikachu-capture.png', frame)

                # Print all the stats to the console
                print(str(num_frames) + '\t' + time + '\t' + str(char1_stock_count) + '\t' + str(char2_stock_count) +
                      '\t' + player1_percentage + '\t' + player2_percentage)

            # Show the video as the analysis is being performed

            if PLAY_VIDEO:
                cv2.imshow('frame', frame)
                if cv2.waitKey(2) & 0xFF == ord('q'):
                    break
        else:
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



# Uses ORB to perform feature matching and returns the
# number of key matches found between the GO template
# and the frame.
# Using training data, it appears the min distance of 10 occurs when GO
# first appears on the screen
def compute_ORB_feature_match_for_start(frame):
    # find the keypoints and descriptors with ORB
    kp2, des2 = ORB.detectAndCompute(frame, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Apply ratio test here to find key matches
    good = []
    for m in matches:
        if m.distance < 11:
            good.append([m])
    return good

# Uses template matching to compute the number of stocks each player has
def compute_num_stocks(player1_stock_image, player2_stock_image, frame):
    # Select the area of the frame that has the player stock information
    frame = frame[STOCK_ROI_UPPER_LEFT [1]: STOCK_ROI_BOTTOM_RIGHT[1], STOCK_ROI_UPPER_LEFT [0]: STOCK_ROI_BOTTOM_RIGHT[0]]

    # Read the character stock template images and grayscale them
    character_template1 = cv2.imread(player1_stock_image, 1)
    character_template1 = cv2.cvtColor(character_template1, cv2.COLOR_BGR2GRAY)
    w1, h1 = character_template1.shape[::-1]
    character_template2 = cv2.imread(player2_stock_image, 1)
    character_template2 = cv2.cvtColor(character_template2, cv2.COLOR_BGR2GRAY)
    w2, h2 = character_template2.shape[::-1]

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform the template matching for each character
    res1 = cv2.matchTemplate(frame, character_template1, cv2.TM_CCOEFF_NORMED)
    res2 = cv2.matchTemplate(frame, character_template2, cv2.TM_CCOEFF_NORMED)

    # Specify a threshold for which the template should match the frame
    threshold = 0.80

    # Store the coordinates of matched area in a numpy array
    loc1 = np.where(res1 >= threshold)
    loc2 = np.where(res2 >= threshold)

    # Sensitivity allows us to determine the number of unique matches
    # and thus the number of stocks
    player1_found = set()
    player2_found = set()
    sensitivity = (w2 * 2) - 1

    for pt in zip(*loc1[::-1]):
        player1_found.add((round(pt[0] / sensitivity), round(pt[1] / sensitivity)))

    for pt in zip(*loc2[::-1]):
        player2_found.add((round(pt[0] / sensitivity), round(pt[1] / sensitivity)))

    return len(player1_found), len(player2_found)

# Compute the player character by performing template matching against
# all player stock icons. Those icons which match the best are deemed to be the player
# character and character variant. Uses two different methodologies to perform the analysis

# One way is to use pure template matching and the other is to use edge detection
def compute_player_character(frame, upper_left, bottom_right, use_canny = False):
    frame = frame[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]

    if use_canny:
        frame = cv2.Canny(frame, 50, 200)

    max_match = 0
    max_character = ''
    # Once the match starts we can use the current frame to determine the characters being played
    # Loop through all character stock icons to determine which one is the best fit
    for filename in os.listdir('data/characters'):
        if not IGNORE_FILENAMES.__contains__(filename):

            # Read in the character stock icon and resize and blur to fit the frame/aspect ration
            character_template = cv2.imread('data/characters/' + filename, 1)
            character_template = cv2.resize(character_template, (0, 0), fx=1.41, fy=1.41)
            character_template = cv2.blur(character_template, (6, 6))

            if use_canny:
                character_template = cv2.Canny(character_template, 50, 200)

            # Perform the template matching and determine if it exceeds the max value thus far
            res = cv2.matchTemplate(frame, character_template, cv2.TM_CCOEFF_NORMED)

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > max_match:
                max_match = max_val
                max_character = filename
    return max_character

# Computes the damage percentage given a player damage percentage ROI
def compute_dmg_percentages(frame, upper_left, bottom_right):
    config = ('-l eng --oem 1 --psm 3')
    frame = frame[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]

    # Make grayscale and invert (black text on white bg) image so that tesseract-OPR can
    # perform better
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.bitwise_not(frame)
    frame = cv2.multiply(frame, 4)
    frame = prepare_for_OCR(frame)

    text = pytesseract.image_to_string(frame, config=config)
    return text

if __name__ == "__main__":
    main()
