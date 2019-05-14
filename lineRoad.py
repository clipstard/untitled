import cv2
import math
import numpy as np
import time
from picamera.array import PiRGBArray
from picamera import PiCamera
import SerialHandler
import threading

serialHandler = SerialHandler.SerialHandler('/dev/ttyACM0')
serialHandler.startReadThread()


def reduce_array(array, target):
    aux = []

    ax1, ay1, ax2, ay2 = target
    for line in array:
        for item in line:
            x1, y1, x2, y2 = item
            if x1 == ax1 and x2 == ax2 and y1 == ay1 and y2 == ay2:
                continue
            else:
                aux.append(item)
    return aux


def find_nearest_left(width, height, lines):
    half_h = height / 2
    half_w = width / 2
    nearest_pack = [0, 0, 0, 0]
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 < half_h and y2 < y1:
            ax1, ay1, ax2, ay1 = nearest_pack
            if x2 > ax2:
                nearest_pack = [x1, y1, x2, y2]

    if nearest_pack == [0, 0, 0, 0]:
        return None
    next_lines = reduce_array(lines, nearest_pack)
    next_nearest = find_next_nearest_left(width, height, next_lines)
    if next_nearest is None or abs(nearest_pack[2] - next_nearest[2]) > 50:
        return [nearest_pack]
    return [nearest_pack, next_nearest]


def find_next_nearest_left(width, height, lines):
    half_h = height / 2
    nearest_pack = [0, 0, 0, 0]
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 < half_h and y2 < y1:
            ax1, ay1, ax2, ay1 = nearest_pack
            if x2 > ax2:
                nearest_pack = [x1, y1, x2, y2]
    if nearest_pack == [0, 0, 0, 0]:
        return None
    return nearest_pack


def find_nearest_right(width, height, lines):
    half_h = height / 2
    half_w = width / 2
    nearest_pack = [100500, 100500, 100500, 100500]
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x1 > half_h and y2 > y1:
            ax1, ay1, ax2, ay1 = nearest_pack
            if x1 < ax1:
                nearest_pack = [x1, y1, x2, y2]

    if nearest_pack == [100500, 100500, 100500, 100500]:
        return None
    next_lines = reduce_array(lines, nearest_pack)
    next_nearest = find_next_nearest_right(width, height, next_lines)
    if next_nearest is None or abs(nearest_pack[0] - next_nearest[0]) > 50:
        return [nearest_pack]

    return [nearest_pack, next_nearest]


def find_next_nearest_right(width, height, lines):
    half_h = height / 2
    half_w = width / 2
    nearest_pack = [100500, 100500, 100500, 100500]
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x1 > half_h and y2 > y1:
            ax1, ay1, ax2, ay1 = nearest_pack
            if x1 < ax1:
                nearest_pack = [x1, y1, x2, y2]

    if nearest_pack == [100500, 100500, 100500, 100500]:
        return None
    return nearest_pack


def make_coordinates(image, line_parameters):
    print('line_params', line_parameters)
    slope, intercept = line_parameters
    if slope == 0:
        return [0, 0, 0, 0]
    print('slope: ', slope, 'intercept: ', intercept)
    y1 = image.shape[0]
    y2 = int((y1 * 2 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    # print('x1: ', x1, 'y1: ', y1, 'x2: ', x2, 'y2: ', y2)

    return np.array([x1, y1, x2, y2])


def print_lines(lines):
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        print ('x1', x1, 'y1', y1, 'x2', x2, 'y2', y2)


def average_slope_intercept(image, lines):
    if lines is None:
        return np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
    image_height, image_width = image.shape[0:2]
    left_found = find_nearest_left(image_width, image_height, lines)
    left_equations = []
    # print(left_found, 'left_found')
    if left_found is not None:
        for x1, y1, x2, y2 in left_found:
            left_equations.append(np.polyfit((x1, x2), (y1, y2), 1))

    right_found = find_nearest_right(image_width, image_height, lines)
    right_equations = []
    # print(right_found, 'right_found')
    if right_found is not None:
        for x1, y1, x2, y2 in right_found:
            right_equations.append(np.polyfit((x1, x2), (y1, y2), 1))

    if not left_equations:
        avg_left = [0, 0]
    else:
        avg_left = np.average(left_equations, axis=0)

    if not right_equations:
        avg_right = [0, 0]
    else:
        avg_right = np.average(right_equations, axis=0)

    right_total = make_coordinates(image, avg_right)
    left_total = make_coordinates(image, avg_left)

    return np.array([left_total, right_total])


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 0, 255)
    return canny


def display_lines(image, lines):
    line_imagez = np.zeros_like(image)
    if lines is not None:
        # for line in lines:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_imagez, (x1, y1), (x2, y2), (255, 0, 0), 20)
    return line_imagez


def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([
        [(-100, height),
         (0+int(width/4),0+int(height/5)),
         (width-int(width/4),0+int(height/5)),
         (width+100, height)]
    ])
    vid = np.array([
        [(0, height), (width, height), (int(width / 2), height-int(height/3))]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    cv2.fillPoly(mask, vid, 0)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
        

def line_compare(line1, line2):
    return None


# image = cv2.imread('ultratest2.png')
# lane_image = np.copy(image)
# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# # cv2.imshow('asda', cropped_image)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 200, np.array([]), maxLineGap=5, minLineLength=30)
# print_lines(lines)
# averaged_lines = average_slope_intercept(lane_image, lines)
#
#
# if averaged_lines is not None:
#     line_image = display_lines(lane_image, averaged_lines)
#
#     combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
#     cv2.imshow("resultq", combo_image)
#     # cv2.imshow("resultqq", combo_image)
# cv2.waitKey(0)

time.sleep(0.1)
def forward(pwm, angle=0.0):
    serialHandler.sendMove(pwm, angle)
    
def stop(angle=0.0):
    serialHandler.sendBrake(angle)
    
default_angle = 19.0
default_large_angle = 23.0
default_speed = 14.0
speed = 15.0
angle = 0.0


last_lines = np.array([[], []])
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('video5.mp4')
camera = PiCamera()
camera.resolution = (800, 600)
camera.framerate = 20
rawCapture = PiRGBArray(camera, size=(800, 600))
moveAllowed = True
for image_frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = image_frame.array
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    cv2.imshow("caa", cropped_image)
    lines = cv2.HoughLinesP(cropped_image, 10, np.pi / 180, 100, np.array([]), maxLineGap=5, minLineLength=40)
    print('mor')
    averaged_lines = average_slope_intercept(frame, lines)
    print(averaged_lines[0], 'avg_lines', averaged_lines[1])
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result", combo_image)
    rawCapture.truncate(0)

    if moveAllowed:
        forward(speed, angle)
    # cv2.imshow("result1", cropped_image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('g'):
        while True:
            c_key = cv2.waitKey(1) & 0xFF
            if c_key == ord('g'):
                break
            time.sleep(1)
    if key == ord('z'):
        break
    if key == ord("w"):
        speed +=1
    if key == ord("s"):
        speed -=1
    if key == ord("a"):
        if angle == 0.0 or angle == -default_large_angle:
            angle = -default_angle
        elif angle == default_angle or default_large_angle:
            angle = 0.0
    if key == ord("d"):
        if angle == 0.0 or angle == default_large_angle:
            angle = default_angle
        elif angle == -default_angle or angle == -default_large_angle:
            angle = 0.0
    if key == ord("q"):
        if angle == 0.0 or angle == -default_angle:
            angle = - default_large_angle
        elif angle == default_angle or angle == default_large_angle:
            angle = 0.0
    if key == ord("e"):
        if angle == 0.0 or angle == default_angle:
            angle = default_large_angle
        elif angle == -default_angle or angle == -default_large_angle:
            angle = 0.0
    if key == ord("r"):
        oldSpeed = -speed
        stop(angle)
        time.sleep(0.5)
        forward(oldSpeed, angle)
        speed = oldSpeed
    if key == ord(" "):
        stop(0.0)
        if moveAllowed ==True:
            moveAllowed = False
        else:
            moveAllowed = True
            if speed < 0:
                speed = -default_speed
            else:
                speed = default_speed
 
cv2.destroyAllWindows()
stop(0.0)

