import cv2
import math
import numpy as np
import time


def reduce_invalid(array, width=640, height=480):
    if array is None:
        return None
    aux = []
    print(width, height, '()')
    for line in array:
        for item in line:
            x1, y1, x2, y2 = item
            if abs(x2 - x1) < 5:
                continue
            elif (x1 > width / 2 and y1 > y2) or (x2 < width / 2 and y2 > y1):
                continue
            else:
                aux.append([[x1, y1, x2, y2]])
    return np.array(aux)


def reduce_horizontals(array):
    if array is None:
        return None
    aux = []
    for line in array:
        # for item in line:
        x1, y1, x2, y2 = line
        if abs(y1 - y2) < 20 and abs(x1 - x2) < 50:
            continue
        else:
            aux.append([x1, y1, x2, y2])
    return np.array(aux)


def reduce_horizontals2(array):
    if array is None:
        return None
    aux = []
    for line in array:
        for item in line:
            x1, y1, x2, y2 = item
            if abs(x1 - x2) > 20 and abs(y1 - y2) < 30 and y1 != y2:
                continue
            else:
                aux.append([[x1, y1, x2, y2]])
    return np.array(aux)


def first_index(x):
    return x[0]


def summarize_lines(lines):
    if lines is None:
        return None
    aux_lines = np.array([first_index(xi) for xi in lines])
    picked_lines = []
    while len(aux_lines):
        if not picked_lines:
            picked_lines.append(aux_lines[0])
            line = picked_lines[-1]
        else:
            line = aux_lines[0]
        nearest_point = find_nearest_point(line, aux_lines)
        if nearest_point is None:
            aux_lines = clear_lines(line, aux_lines)
            picked_lines = safe_append(line, picked_lines)
        while nearest_point is not None:
            line = add_lines(line, nearest_point)
            picked_lines[-1] = line
            aux_lines = clear_lines(nearest_point, aux_lines)
            nearest_point = find_nearest_point(line, aux_lines)
    return picked_lines


def add_lines(line1, line2):
    ax1, ay1, ax2, ay2 = line1
    bx1, by1, bx2, by2 = line2
    return [ax1, ay1, bx2, by2]


def safe_append(line, lines):
    aux_lines = lines
    for a_line in lines:
        if lines_are_equal(line, a_line):
            return lines
    x1, y1, x2, y2 = line
    aux_lines.append([x1, y1, x2, y2])
    return aux_lines


def clear_lines(line, lines):
    final_lines = []
    for a_line in lines:
        if lines_are_equal(line, a_line):
            continue
        final_lines.append(a_line)
    return final_lines


def lines_are_equal(line1, line2):
    ax1, ay1, ax2, ay2 = line1
    bx1, by1, bx2, by2 = line2
    return ax1 == bx1 and ay1 == by1 and ax2 == bx2 and ay2 == by2


def find_nearest_point(line, lines):
    ax1, ay1, ax2, ay2 = line
    for l in lines:
        x1, y1, x2, y2 = l
        if abs(x1 - ax2) < 10 and abs(y1 - ay2) < 10:
            return l
    return None


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    if slope == 0:
        return [0, 0, 0, 0]
    print('slope: ', slope, 'intercept: ', intercept)
    y1 = image.shape[0]
    y2 = int((y1 * 2 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])


def print_lines(lines):
    count = 0
    for line in lines:
        count += 1
        x1, y1, x2, y2 = line
        if x2 > x1:
            print ('x1', x1, 'y1', y1, 'x2', x2, 'y2', y2, 'x2 > x1')
        elif x1 > x2:
            print ('x1', x1, 'y1', y1, 'x2', x2, 'y2', y2, 'x2 < x1')
        elif y1 > y2:
            print ('x1', x1, 'y1', y1, 'x2', x2, 'y2', y2, 'y2 < y1')
        else:
            print ('x1', x1, 'y1', y1, 'x2', x2, 'y2', y2, 'other')


def make_average_coordinates(lines):
    if lines is None or not len(lines):
        return None
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    count = 0
    for line in lines:
        ax1, ay1, ax2, ay2 = line
        x1 += ax1
        x2 += ax2
        y1 += ay1
        y2 += ay2
        count += 1

    if count == 0:
        return None
    line = [x1/count, y1/count, x2/count, y2/count]
    return np.array([line])


def is_order(array, limit):
    counter = 1
    last_item = array[0]
    for item in array:
        if item > last_item:
            counter += 1
        else:
            counter = 1
        last_item = item
        if counter == limit:
            return True
    return False


def line_is_interrupted(lines):
    aux = []
    for line in lines:
        x1, y1, x2, y2 = line
        aux.append(abs(y1 - y2))
    return is_order(aux, 3)


def average_slope_intercept(image, lines):
    print(lines)
    if lines is None:
        return [None, [None, None]]  # for left_interrupted / right interrupted error

    left_fit = []
    right_fit = []
    stop_fit = []
    left_total = []
    right_total = []
    stop_total = []
    left_lines = []
    right_lines = []
    left_is_interrupted = False
    right_is_interrupted = False
    lines = reduce_horizontals2(lines)
    summarized_lines = summarize_lines(lines)
    summarized_lines = reduce_horizontals(summarized_lines)
    for line in summarized_lines:
        x1, y1, x2, y2 = line.reshape(4)
        print(x1, y1, x2, y2)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = parameters[0:2]
        if y1 < y2:
            left_fit.append((slope, intercept))
            left_lines.append([x1, y1, x2, y2])
        elif y2 < y1:
            right_fit.append((slope, intercept))
            right_lines.append([x1, y1, x2, y2])
        else:
            stop_fit.append([x1, y1, x2, y2])

    if len(left_fit):
        left_fit_avg = np.average(left_fit, axis=0)
        left_total = make_coordinates(image, left_fit_avg)
        left_is_interrupted = line_is_interrupted(left_lines)

    if len(right_fit):
        right_fit_avg = np.average(right_fit, axis=0)
        right_total = make_coordinates(image, right_fit_avg)
        right_is_interrupted = line_is_interrupted(right_lines)

    if len(stop_fit):
        stop_total = make_average_coordinates(stop_fit)

    results = []
    if len(left_total):
        results.append(left_total)
    else:
        results.append(None)
    if len(right_total):
        results.append(right_total)
    else:
        results.append(None)
    if stop_total is not None and len(stop_total):
        results.append(stop_total)

    print(results)
    if not len(results):
        return [None, None]
    else:
        return [np.array(results), [left_is_interrupted, right_is_interrupted]]


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 0, 255)
    return canny


def display_lines(image, lines):
    line_imagez = np.zeros_like(image)
    count = 3
    colors = [
        (255, 0, 0),
        (100, 100, 100),
        (0, 0, 0),
        (255, 255, 255),
        (0, 255, 0),
        (255, 255, 0),
        (0, 0, 255),
        (0, 255, 255),
        (255, 0, 255),
        (100, 255, 0),
        (255, 100, 0),
        (255, 0, 100),
        (100, 0, 255),
        (255, 100, 255),
        (100, 100, 0),
        (255, 255, 100),
        (100, 100, 255),
        (100, 0, 100),
        (50, 100, 255),
        (50, 50, 50),
        (255, 100, 50),
        (50, 100, 0),
        (255, 50, 0),
        (255, 0, 50),
        (255, 50, 50),
        (100, 100, 50),
        (50, 100, 100),
        (0, 50, 0),
        (0, 0, 50)
    ]

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            count += 1
            if count > len(colors) - 1:
                count = count - len(colors)
            cv2.line(line_imagez, (x1, y1), (x2, y2), colors[count], 10)
            print(x1, y1, x2, y2, colors[count], 'color')
    return line_imagez


def display_average_lines(image, lines, interrupted):
    line_imagez = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line.reshape(4)
                if y1 < y2 and interrupted[1] or y1 > y2 and interrupted[0]:
                    cv2.line(line_imagez, (x1, y1), (x2, y2), (0, 255, 0), 10)
                else:
                    cv2.line(line_imagez, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_imagez


def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    print(width, height)
    polygons = np.array([
        [(-100, height),
         (0 + int(width / 4), 0 + int(height / 3)),
         (width - int(width / 4), 0 + int(height / 3)),
         (width + 100, height)]
    ])
    vid = np.array([
        [(0, height),
         (width, height),
         (int(width / 2), height - int(height / 3))]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    cv2.fillPoly(mask, vid, 0)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


# image = cv2.imread('img/supertest.jpg')
# # image = cv2.imread('img/30.png')
# lane_image = np.copy(image)
# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# cv2.imshow('asda', cropped_image)
# lines = cv2.HoughLinesP(cropped_image, 2, (np.pi / 180), 100, np.array([]), minLineLength=20, maxLineGap=10)
#
# lines = reduce_invalid(lines)
# lines = reduce_horizontals2(lines)
#
# # print_lines(lines)
# averaged_lines, lines_interrupted = average_slope_intercept(cropped_image, lines)
#
# left_is_interrupted, right_is_interrupted = lines_interrupted
#
# if averaged_lines is not None:
#     line_image = display_average_lines(lane_image, averaged_lines, lines_interrupted)
#     combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
#     cv2.imshow("resultq", combo_image)
#     # cv2.imshow("resultqq", combo_image)
# cv2.waitKey(0)

# afisarea imaginii in axele x si y, pentru a determina virfurile la triunghiul de detectarea liniilor,
# pentru region_of_interest:
#

last_lines = []
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('video5.mp4')
while (cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, (np.pi / 180), 100, np.array([]), minLineLength=20, maxLineGap=10)
    height = frame.shape[1]
    width = frame.shape[0]
    lines = reduce_invalid(lines, height, width)
    lines = reduce_horizontals2(lines)
    averaged_lines, lines_interrupted = average_slope_intercept(cropped_image, lines)
    if averaged_lines is not None:
        line_image = display_average_lines(frame, averaged_lines, lines_interrupted)
        # line_image = display_lines(frame, lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    # cv2.imshow("masked", cropped_image)
        cv2.imshow("result", combo_image)
    else:
        cv2.imshow("result", frame)
    # cv2.imshow("result1", cropped_image)
    key = cv2.waitKey(1)
    if key == ord('g'):
        while True:
            c_key = cv2.waitKey(1)
            if c_key == ord('g'):
                break
            time.sleep(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
