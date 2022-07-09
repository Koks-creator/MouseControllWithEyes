import mediapipe as mp
import cv2
import numpy as np
import math
import autopy
import mouse
from time import sleep
from collections import deque
import json
import os
import pyglet
from threading import Thread

click_file = "klik.mp3"
start_cal_file = "elo.wav"
end_cal_file = "elo2.wav"


if os.path.isfile('config_mouse.json'):
    with open("config_mouse.json", 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
    thresh_lconf = json_object['Thresh_l']
    thresh_rconf = json_object['Thresh_r']

else:
    config_data = {
        "Thresh_l": 155,
        "Thresh_r": 60,
    }
    json_object = json.dumps(config_data, indent=4)
    config_file = open("config.json", "w+")
    config_file.write(json_object)
    config_file.close()

    thresh_lconf = 155
    thresh_rconf = 60


def play_sound(file):
    current_path = os.getcwd() + "\\"
    sound = pyglet.media.load(current_path + file)
    sound.play()


def get_bbox():
    x = lm_list[123][0]
    y = lm_list[10][1]
    w = lm_list[352][0] - lm_list[123][0]
    h = lm_list[152][1] - lm_list[10][1]

    return x, y, w, h


def calibrate_eyes(gaze_r):
    global l_t_eye
    global r_t_eye
    global left_good_res
    global right_good_res
    global start_calibration
    global left_cal
    global right_cal
    global final_cal_left
    global final_cal_right
    global calibrated_list

    if left_cal is False:
        cv2.circle(img, (int(width / 2), int(height / 2) - 50), 30, (255, 255, 0), -1)
        if 1.005 < gaze_r < 1.3:
            left_good_res += 1

            if left_good_res == 3:
                left_good_res = 0
                l_t_eye = l_t_eye
                left_cal = True
                final_cal_left = l_t_eye
                cv2.setTrackbarPos("Thresh_l", "Options", 0)
                l_t_eye = 200
                calibrated_list.append(True)
                print("spoko lewy")
        else:
            l_t_eye -= 2
            sleep(.25)
            cv2.setTrackbarPos("Thresh_l", "Options", l_t_eye)

    if left_cal:
        if right_cal is False:
            cv2.circle(img, (int(width / 2), int(height / 2) - 50), 30, (255, 50, 100), -1)
            if 1.005 < gaze_r < 1.3:
                right_good_res += 1

                if right_good_res == 3:
                    right_good_res = 0
                    r_t_eye = r_t_eye
                    right_cal = True
                    final_cal_right = r_t_eye
                    # cv2.setTrackbarPos("Thresh_r", "Options", 50)
                    cv2.setTrackbarPos("Thresh_r", "Options", final_cal_right)
                    cv2.setTrackbarPos("Thresh_l", "Options", final_cal_left)
                    r_t_eye = 50
                    calibrated_list.append(True)
                    print("spoko prawy")
                    start_calibration = False
                    left_cal = False
                    right_cal = False
                    Thread(target=play_sound, args=(end_cal_file, )).start()
                    calibrated_list.clear()
            else:
                r_t_eye += 2
                sleep(.25)
                cv2.setTrackbarPos("Thresh_r", "Options", r_t_eye)


ratio_list = deque(maxlen=3)


def check_if_blinking(points: list):
    global left_blink
    global right_blink
    global vertical_direction

    cv2.circle(img, lm_list[points[0]], 5, (255, 0, 0), -1)
    cv2.circle(img, lm_list[points[1]], 5, (255, 0, 0), -1)

    cv2.circle(img, lm_list[points[2]], 5, (255, 0, 0), -1)
    cv2.circle(img, lm_list[points[3]], 5, (255, 0, 0), -1)

    cv2.line(img, lm_list[points[0]], lm_list[points[2]], (0, 255, 255), 4)
    cv2.line(img, lm_list[points[0]], lm_list[points[3]], (0, 255, 255), 4)
    cv2.line(img, lm_list[points[1]], lm_list[points[2]], (0, 255, 255), 4)
    cv2.line(img, lm_list[points[1]], lm_list[points[3]], (0, 255, 255), 4)

    left_up = lm_list[points[0]]
    left_down = lm_list[points[1]]
    left_left = lm_list[points[2]]
    left_right = lm_list[points[3]]

    length_ver = math.hypot(left_up[0] - left_down[0], left_up[1] - left_down[1])
    length_hor = math.hypot(left_left[0] - left_right[0], left_left[1] - left_right[1])

    ratio = (length_ver / length_hor) * 100
    ratio_list.append(ratio)
    ratio_avg = sum(ratio_list) / len(ratio_list)
    # print(blink_ratio)
    # cv2.putText(img, f"Blink ratio: {blink_ratio}", (10, 200), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)

    if vertical_direction == "Center":
        if points[0] == 159:
            if ratio_avg < 15:
                left_blink = "Blinking"
            else:
                left_blink = ""
        else:
            if ratio_avg < 15:
                right_blink = "Blinking"
            else:
                right_blink = ""

        return ratio_avg


def get_landmarks(img: np.array) -> list:
    lm_list = []
    h, w, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results_object = face_mesh.process(img_rgb)
    results = results_object.multi_face_landmarks
    if results:
        for landmark in results:
            for lm in landmark.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)

                lm_list.append((x, y))

    return lm_list


def get_gaze_ratio(points, t):
    eye_region = np.array([lm_list[points[0]], lm_list[points[1]], lm_list[points[2]], lm_list[points[3]],
                           lm_list[points[4]], lm_list[points[5]]], np.int32)


    # cv2.polylines(img, [left_eye_region], True, (255, 80, 0), 2)

    height, width, _ = img.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray_img, gray_img, mask=mask)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    # Wycinanie oka
    gray_eye = eye[min_y:max_y, min_x:max_x]

    _, threshold_eye = cv2.threshold(gray_eye, t, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape

    whole_eye_white = cv2.countNonZero(threshold_eye)

    # Podzielenie na pol
    left_side_threshold = threshold_eye[0:height, 0:int(width / 2)]
    # liczenie liczny pikseli zrenicy
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0:height, int(width / 2):width]
    # liczenie liczny pikseli zrenicy
    right_side_white = cv2.countNonZero(right_side_threshold)

    # podzielenie na pol, ale inaczej
    top_side_threshold = threshold_eye[0:int(height / 4), 0:width]
    top_side_white = cv2.countNonZero(top_side_threshold)

    bottom_side_threshold = threshold_eye[int(height / 1.25):height, 0:width]
    bottom_side_white = cv2.countNonZero(bottom_side_threshold)

    # gdzie jest wiecej zrenicy, czy na lewo czy na prawo
    gaze_ratio = left_side_white / (right_side_white + 0.000001)

    # gdzie jest wiecej zrenicy, czy na gorze czy na dole
    gaze_ratio2 = bottom_side_white / (top_side_white + 0.00001)

    return gaze_ratio, gaze_ratio2, top_side_threshold, bottom_side_threshold, threshold_eye, whole_eye_white


def angle2pt(a, b):
    change_inx = b[0] - a[0]
    change_iny = b[1] - a[1]
    ang = math.degrees(math.atan2(change_iny, change_inx))
    return ang


def nothing(x):
    pass


cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("video.mp4")
face = mp.solutions.face_mesh
face_mesh = face.FaceMesh()

cv2.namedWindow("Options")
cv2.resizeWindow("Options", 400, 400)
cv2.createTrackbar("Thresh_l", "Options", thresh_lconf, 255, nothing)
cv2.createTrackbar("Thresh_r", "Options", thresh_rconf, 255, nothing)

# do kalibracji
left_good_res = 0
right_good_res = 0
start_calibration = False
left_cal = False
right_cal = False
calibrated_list = []
final_cal_left = 0
final_cal_right = 0
l_t_eye = 200
r_t_eye = 50
calibrate_frames = 0
#######################

# do kierunkow
vertical_direction = ""
horizontal_direction = ""
p_ver_dir = ""
p_hor_dir = ""
dir_frames = 0
p_dir_frames = 0
directions_history = deque(maxlen=10)
#####################

# do kulki co sie rusza
kulka_x, kulka_y = 600, 300
kolerek_kulki = (255, 0, 255)
cursor_step = 10
#####################
# Blinking
left_blink = ""
right_blink = ""
p_left_blink = ""
p_right_blink = ""
blinking_frames = 0

# Klikanie
left_click_frames = 0
right_click_frames = 0

# Reszta
x1_border = 200
border_width = 880
y1_border = 60
border_height = 600

x1_border2 = x1_border + 300
x2_border2 = x1_border + border_width - 300
w_scr, h_scr = autopy.screen.size()

white_pixels_history_l = deque(maxlen=15)
white_pixels_history_r = deque(maxlen=15)
white_pixels_history_l2 = deque(maxlen=15)
white_pixels_history_r2 = deque(maxlen=15)

calibration_frames = 0
while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    thresh_l = cv2.getTrackbarPos("Thresh_l", "Options")
    thresh_r = cv2.getTrackbarPos("Thresh_r", "Options")
    success, img = cap.read()
    if success is False:
        break

    img = cv2.flip(img, 1)
    img = cv2.resize(img, (1280, 720))
    height, width, _ = img.shape
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lm_list = get_landmarks(img)
    if len(lm_list) != 0:
        # lewo, prawo
        cv2.line(img, (10, 200), (10, 500), (0, 0, 200), 50)
        cv2.line(img, (1270, 200), (1270, 500), (0, 0, 200), 50)
        # gora, dol
        cv2.line(img, (440, 710), (840, 710), (0, 0, 200), 50)
        cv2.line(img, (440, 10), (840, 10), (0, 0, 200), 50)

        # ROI cale te (duzy kwadrat) gdzie myszkuje sie
        cv2.rectangle(img, (x1_border, y1_border), (x1_border + border_width, y1_border + border_height), (0, 0, 200), 4)
        # maly kwadrat, zeby zacislic
        cv2.rectangle(img, (x1_border2, 260), (x2_border2, 460), (0, 0, 200), 4)
        # tu na oczy
        cv2.rectangle(img, (x1_border2, 260), (x2_border2, 320), (0, 0, 200), 4)

        # Granice scrollowania
        cv2.line(img, (x1_border2 + 20, 550), (x2_border2 - 20, 550), (255, 0, 255), 4)
        cv2.line(img, (x1_border2 + 20, 170), (x2_border2 - 20, 170), (255, 0, 255), 4)

        bbox_x, bbox_y, bbox_w, bbox_h = get_bbox()

        cv2.rectangle(img, (bbox_x, bbox_y), (bbox_x + 80, bbox_y - 40), (0, 200, 50), -1)
        cv2.rectangle(img, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (0, 200, 50), 4)
        cv2.putText(img, f"{bbox_w}", (bbox_x + 10, bbox_y - 8), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 4)
        if 300 < bbox_w < 450:

            # kat glowy, zeby za bardzo nie odchylac
            bottom_face = lm_list[152]
            top_face = lm_list[10]

            ang = math.degrees(math.atan2(top_face[0] - bottom_face[0], top_face[1] - bottom_face[1]))
            ang = ang * -1 if ang < 0 else ang
            cv2.circle(img, lm_list[6], 10, (255, 0, 0), -1)

            if x1_border2 < lm_list[6][0] < x2_border2 and 260 < lm_list[6][1] < 320 and 172 < ang < 181:
                x2 = np.interp(kulka_x, (x1_border, x1_border + border_width), (0, w_scr))
                y2 = np.interp(kulka_y, (y1_border, y1_border + border_height), (0, h_scr))

                points = [33, 160, 158, 173, 153, 144]
                points2 = [463, 384, 386, 467, 373, 380]

                left_gaze_ratio, left_gaze_ratio2, left_top_side_threshold, left_bottom_side_threshold, left_threshold_eye, \
                left_white = get_gaze_ratio(points, thresh_l)
                right_gaze_ratio, right_gaze_ratio2, right_top_side_threshold, right_bottom_side_threshold, right_threshold_eye,\
                right_white = get_gaze_ratio(points2, thresh_r)

                #  Dynamiczna korekcja

                # jak za duzy threshold
                # print(start_calibration)
                if start_calibration is False:

                    # print(left_white)
                    if left_white < 700:
                        white_pixels_history_l.append(True)
                        if all(white_pixels_history_l) and len(white_pixels_history_l) == 15:
                            thresh_l -= 5
                            cv2.setTrackbarPos("Thresh_l", "Options", thresh_l)
                            # print(left_white)
                    else:
                        white_pixels_history_l.append(False)
                    if right_white < 700:
                        white_pixels_history_r.append(True)
                        if all(white_pixels_history_r) and len(white_pixels_history_r) == 15:
                            # print("xcvghbj")
                            thresh_r -= 5
                            cv2.setTrackbarPos("Thresh_r", "Options", thresh_r)
                            # print(left_white)
                    else:
                        white_pixels_history_r.append(False)

                    # jak za maly threshold
                    if left_white > 800:
                        white_pixels_history_l2.append(True)
                        if all(white_pixels_history_l2) and len(white_pixels_history_l2) == 15:
                            # print("xcvghbj")
                            thresh_l += 5
                            cv2.setTrackbarPos("Thresh_l", "Options", thresh_l)
                            # print(left_white)
                    else:
                        white_pixels_history_l2.append(False)

                    if right_white > 800:
                        white_pixels_history_r2.append(True)
                        if all(white_pixels_history_r2) and len(white_pixels_history_r2) == 15:
                            # print("xcvghbj")
                            thresh_r += 5
                            cv2.setTrackbarPos("Thresh_r", "Options", thresh_r)
                            # print(left_white)
                    else:
                        white_pixels_history_r2.append(False)

                cv2.imshow("left_threshold_eye", left_threshold_eye)
                cv2.imshow("right_threshold_eye", right_threshold_eye)

                gaze_ratio_left_right = (left_gaze_ratio + right_gaze_ratio) / 2
                gaze_ratio_up_down = (left_gaze_ratio2 + right_gaze_ratio2) / 2

                left_blink_ratio = check_if_blinking([159, 145, 33, 133])
                right_blink_ratio = check_if_blinking([374, 386, 362, 359])

                cv2.putText(img, f"l_blink_ratio: {round(left_blink_ratio, 3) if left_blink_ratio is not None else left_blink_ratio}",
                            (40, 400), cv2.FONT_HERSHEY_PLAIN, 2.7, (0, 0, 255), 4)
                cv2.putText(img, f"r_blink_ratio: {round(right_blink_ratio, 3) if right_blink_ratio is not None else right_blink_ratio}",
                            (40, 500), cv2.FONT_HERSHEY_PLAIN, 2.7, (0, 0, 255), 4)

                # granice kierunkow, poruszanie myszki
                if left_blink != "Blinking" or right_blink != "Blinking":
                    if any(directions_history) is False:  # zeby bylo opoznienie po klikknieciu
                        if horizontal_direction == "Center" or horizontal_direction == "":
                            if gaze_ratio_up_down > 3.5:
                                vertical_direction = "Up"
                                if kulka_y - y1_border > cursor_step:
                                    if vertical_direction == p_ver_dir:
                                        kulka_y -= 10
                            elif gaze_ratio_up_down < 1.2 and left_blink != "Blinking" and right_blink != "Blinking":
                                vertical_direction = "Down"
                                if (y1_border + border_height) - kulka_y > cursor_step:
                                    if vertical_direction == p_ver_dir:
                                        kulka_y += 10
                            else:
                                vertical_direction = "Center"

                        if vertical_direction == "Center" or vertical_direction == "":
                            if gaze_ratio_left_right < .9:
                                horizontal_direction = "Left"
                                if kulka_x - x1_border > cursor_step:
                                    if vertical_direction == p_ver_dir:
                                        kulka_x -= 10
                            elif .9 <= gaze_ratio_left_right < 1.9:
                                horizontal_direction = "Center"
                            else:
                                horizontal_direction = "Right"
                                if (x1_border + border_width) - kulka_x > cursor_step:
                                    if vertical_direction == p_ver_dir:
                                        kulka_x += 10
                        # if vertical_direction == p_ver_dir: #stestowac
                        try:
                            autopy.mouse.move(x2, y2)
                        except Exception as e:
                            print(e)
                # print(vertical_direction)

                # Scrollowanie
                if x1_border2 < kulka_x < x2_border2 and 550 < kulka_y < 630:
                    mouse.wheel(-.5)

                if x1_border2 < kulka_x < x2_border2 and 170 > kulka_y > 50:
                    mouse.wheel(.5)

                # klikanie
                if vertical_direction == "Center" and horizontal_direction == "Center":
                    if right_blink == "Blinking":
                        left_click_frames += 1
                        calibration_frames += 1
                        if left_click_frames == 4:
                            directions_history.append(True)
                            print("left")
                            mouse.click('left')
                            Thread(target=play_sound, args=(click_file,)).start()
                            kolerek_kulki = (0, 0, 200)
                            left_click_frames = 0
                    else:
                        calibration_frames = 0
                        directions_history.append(False)
                        kolerek_kulki = (255, 0, 255)
                        left_click_frames = 0

                    if left_blink == "Blinking" and right_blink == "":
                        right_click_frames += 1
                        if right_click_frames == 3:
                            directions_history.append(True)
                            print("right")
                            # mouse.click('right')
                            Thread(target=play_sound, args=(click_file,)).start()
                            kolerek_kulki = (0, 0, 200)
                            right_click_frames = 0
                    else:
                        directions_history.append(False)
                        kolerek_kulki = (255, 0, 255)
                        right_click_frames = 0

                # print(calibration_frames)
                if calibration_frames >= 40:
                    Thread(target=play_sound, args=(start_cal_file,)).start()
                    cv2.setTrackbarPos("Thresh_r", "Options", 0)
                    cv2.setTrackbarPos("Thresh_l", "Options", 200)
                    start_calibration = True
                    calibration_frames = 0
                # kalibracja
                if start_calibration is True:
                    cv2.putText(img, f"Calibrating...", (800, 80), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)
                    cv2.putText(img, f"Keep staring at point", (390, 370), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 4)
                    cv2.putText(img, f"Eyes calibrated {len(calibrated_list)}/2", (300, 200), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 4)
                    calibrate_eyes(gaze_ratio_left_right)
                else:
                    cv2.circle(img, (int(width / 2), int(height / 2)), 30, (255, 255, 0), -1)
                    # cv2.putText(img, f"bbox_w: {bbox_w}", (10, 500), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)
                    cv2.putText(img, f"Left blinking: {left_blink}", (40, 60), cv2.FONT_HERSHEY_PLAIN, 2.2, (100, 40, 200), 4)
                    cv2.putText(img, f"Right blinking: {right_blink}", (40, 120), cv2.FONT_HERSHEY_PLAIN, 2.2, (100, 40, 200), 4)
                    cv2.putText(img, f"Ver dir: {vertical_direction}({round(gaze_ratio_up_down, 2)})", (40, 180),
                                cv2.FONT_HERSHEY_PLAIN, 2.2, (100, 40, 200), 4)
                    cv2.putText(img, f"Hor dir: {horizontal_direction}({round(gaze_ratio_left_right, 2)})", (40, 240),
                                cv2.FONT_HERSHEY_PLAIN, 2.2, (100, 40, 200), 4)

                    p_ver_dir = vertical_direction
                    p_hor_dir = horizontal_direction
                    p_left_blink = left_blink
                    p_right_blink = right_blink

                cv2.circle(img, (kulka_x, kulka_y), 15, kolerek_kulki, -1)
            else:
                # print(ang)
                cv2.putText(img, f"Keep your head straight", (320, 200), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 4)
                # Thread(target=play_sound, args=(start_cal_file,)).start()
        else:
            cv2.putText(img, f"You're too close/too far", (250, 500), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 4)

    cv2.imshow("res", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

    if key == ord("s"):
        if key == ord('s'):
            config = {
                "Thresh_l": thresh_l,
                "Thresh_r": thresh_r,
            }
            json_object = json.dumps(config, indent=4)
            config_file = open("config_mouse.json", "w+")
            config_file.write(json_object)
            config_file.close()
            print("Config saved...")
            print("New configuration will be applied after next run.")

    if key == ord('c'):
        Thread(target=play_sound, args=(start_cal_file,)).start()
        cv2.setTrackbarPos("Thresh_r", "Options", 0)
        cv2.setTrackbarPos("Thresh_l", "Options", 200)
        start_calibration = True
