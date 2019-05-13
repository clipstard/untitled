import cv2
import math
import numpy as np
import time
import constant
import threading
from picamera.array import PiRGBArray
from picamera import PiCamera

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(0.1)
listen_to_lines = True
const_actions = [
    constant.RIGHT,
    constant.LEFT,
    constant.FORWARD,
    constant.PARKING,
    constant.RIGHT,
    constant.LEFT,
    constant.RIGHT,
    constant.LEFT,
    constant.FINISH
]


def wait(seconds):
    seconds = float(seconds)
    time.sleep(seconds)


def custom_wait(seconds):
    now = time.time()
    while time.time() < now + seconds:
        pass

def forward(speed, angle):
    print("I\'m moving forward with %d speed and %d angle" % (speed, angle))


def move_in_intersection(direction):
    global listen_to_lines
    listen_to_lines = False
    if direction == constant.LEFT:
        wait(1.0)
        listen_to_lines = True
    elif direction == constant.RIGHT:
        wait(1.0)
        listen_to_lines = True
    elif direction == constant.FORWARD:
        wait(1.0)
        listen_to_lines = True
    elif direction == constant.STOP:
        for i in range(0, 3):
            print("%d," % i+1)
            wait(1.0)
        print('go')
        listen_to_lines = True
    else:
        wait(4.0)
        listen_to_lines = True
    return True


def de_array(array):
    aux = []
    if array is None:
        return None
    if isinstance(array, list):
        items = []
        if len(array) == 1:
            return de_array(array[0])
        for item in array:
            if item is None:
                items.append(None)
            elif isinstance(item, list):
                items.append(de_array(item))
            else:
                return array
        return items
    return aux


def normalize_line(line):
    if isinstance(line, list):
        if len(line) == 4:
            return [line]
        if len(line) == 1 and isinstance(line[0], list):
            inner_line = line[0]
            if len(inner_line) == 4:
                return [inner_line]
            if len(inner_line) == 1 and isinstance(inner_line[0], list):
                second_line = inner_line[0]
                if len(second_line) == 4:
                    return [second_line]
                else:
                    return second_line
    return line


def make_sum(array):
    sum = 0
    if array is None:
        return 0
    for item in array:
        sum += item
    return sum


def calculate_angle(lines):
    if lines is None:
        return 0
    angles = []

    for line in lines:
        if line is not None:
            line = normalize_line(line)
            for x in line:
                if x is None:
                    continue
                x1, y1, x2, y2 = x
                if abs(y1 - y2) < 15:
                    continue
                just = float(x2 - x1) / float(y1 - y2)
                angles.append(just)
    return make_sum(angles)


def reduce_invalid(array, width=640, height=480):
    if array is None:
        return None
    aux = []
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
    result = []
    for x in picked_lines:
        result.append([x])
    return np.array(result)


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
    lines = de_array(lines)
    if isinstance(lines, list) and len(lines) == 4 and not isinstance(lines[0], list):
        lines = normalize_line(lines)
    print(lines, 'incoming lines')
    for line in lines:
        line = normalize_line(line)
        if line is None:
            continue
        if isinstance(line, list):
            x = line[0]
            if x is None:
                continue
            ax1, ay1, ax2, ay2 = x
        else:
            ax1, ay1, ax2, ay2 = line
        x1 += ax1
        x2 += ax2
        y1 += ay1
        y2 += ay2
        count += 1

    if count == 0:
        return None
    line = [x1 / count, y1 / count, x2 / count, y2 / count]
    return [line]


def is_order(array, limit, value=45, max_larges=1):
    counter = 1
    last_item = array[0]
    count_larges = 0
    for item in array:
        if item > value:
            count_larges += 1
        if count_larges >= max_larges:
            return False
        if item > last_item:
            counter += 1
        else:
            counter = 1
        last_item = item
        if counter == limit:
            return True
    return len(array) > 2 and count_larges < max_larges


def line_is_interrupted(lines):
    aux = []
    if len(lines) < 3:
        return False
    for line in lines:
        x1, y1, x2, y2 = line
        aux.append(abs(y1 - y2))
    return is_order(aux, 3, 60, 2)


def average_slope_intercept(image, lines):
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
    summarized_lines = summarize_lines(lines)
    summarized_lines = reduce_horizontals2(summarized_lines)
    for line in summarized_lines:
        x1, y1, x2, y2 = line.reshape(4)
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
    return line_imagez


def display_average_lines(image, lines, interrupted):
    line_imagez = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            line = normalize_line(line)
            if line is not None:
                if isinstance(line, list):
                    x = line[0]
                    x1, y1, x2, y2 = x
                else:
                    x1, y1, x2, y2 = line.reshape(4)
                if y1 < y2 and interrupted[1] or y1 > y2 and interrupted[0]:
                    cv2.line(line_imagez, (x1, y1), (x2, y2), (0, 255, 0), 10)
                else:
                    cv2.line(line_imagez, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_imagez


def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
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


def convert_numpy_to_array(numpy_array):
    aux = []
    if numpy_array is None:
        return None
    if isinstance(numpy_array, np.ndarray) and numpy_array.size == 1 and numpy_array.item(0) is None:
        return None
    if isinstance(numpy_array, np.ndarray) and numpy_array.size == 4:
        x1, y1, x2, y2 = numpy_array
        return normalize_line([x1, y1, x2, y2])
    for item in numpy_array:
        if item is None:
            aux.append([None])
            continue
        if isinstance(item, list):
            item = normalize_line(item)
            x = item[0]
            x1, y1, x2, y2 = x
        else:
            x1, y1, x2, y2 = item.reshape(4)
        aux.append([[x1, y1, x2, y2]])
    return aux


def make_average_lines(last_lines, current_lines):
    aux = []
    if last_lines is None or not len(last_lines):
        print('issues')
        return current_lines
    if len(last_lines) < len(current_lines):
        if isinstance(last_lines, list):
            last_lines.append(None)
        else:
            last_lines = convert_numpy_to_array(last_lines)
            last_lines.append(None)
            last_lines = np.array(last_lines)
    for i in range(0, len(current_lines)):
        print(i, 'i')
        if current_lines[i] is None and last_lines[i] is not None:
            aux.append(normalize_line(convert_numpy_to_array(last_lines[i])))
        elif current_lines[i] is not None and last_lines[i] is None:
            aux.append(normalize_line(convert_numpy_to_array(current_lines[i])))
        else:
            print(last_lines[i], current_lines[i], 'with i')
            print(last_lines, current_lines, 'yoyoyo')
            aux.append(normalize_line(convert_numpy_to_array(make_average_coordinates([last_lines[i], current_lines[i]]))))
    return np.array(de_array(aux))


def equation(previous_value):
    max = 23.0
    if previous_value < -10:
        return previous_value / 2.5 + 1
    if previous_value < 0:
        return previous_value / 1.5 + 1
    if previous_value < 5:
        return previous_value + 2.5
    else:
        if previous_value < 10:
            num = previous_value * 1.5 + 1
        else:
            num = previous_value * 2.5 + 1
        if num > max:
            return max
        return num


def down_equation(previous_value):
    min = -23.0
    if previous_value > 10:
        return previous_value / 2.5 - 1
    if previous_value > 0:
        return float(previous_value / 1.5 - 1)
    if previous_value > -5:
        return previous_value - 2.5
    else:
        if previous_value > -10:
            num = previous_value * 1.5 - 1
        else:
            num = previous_value * 2.5 - 1
        if num < min:
            return min
        return num


last_lines = []
speed = 15.0
angle = 0.0
count = 0
decision = 0.0
for camera_frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = camera_frame.array
    if listen_to_lines:
        canny_image = canny(frame)
        cropped_image = region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_image, 2, (np.pi / 180), 100, np.array([]), minLineLength=20, maxLineGap=10)
        height = frame.shape[1]
        width = frame.shape[0]
        lines = reduce_invalid(lines, height, width)
        lines = reduce_horizontals2(lines)
        to_check_lines = None
        averaged_lines, lines_interrupted = average_slope_intercept(cropped_image, lines)
        if averaged_lines is not None:
            to_check_lines = make_average_lines(last_lines, averaged_lines)
            last_lines = averaged_lines
            line_image = display_average_lines(frame, to_check_lines, lines_interrupted)
            combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
            cv2.imshow("result", combo_image)
        else:
            if count < 2:
                to_check_lines = last_lines
                line_image = display_average_lines(frame, to_check_lines, lines_interrupted)
                # calculate_angle(to_check_lines)
                combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
                cv2.imshow("result", combo_image)
                count += 1
            else:
                cv2.imshow("result", frame)
        if to_check_lines is not None:
            calculated_angle = calculate_angle(convert_numpy_to_array(to_check_lines))
            print(calculated_angle)
            if calculated_angle > 0:
                decision = equation(decision)
            else:
                decision = down_equation(decision)

        key = cv2.waitKey(1)
        if key == ord('p'):
            x = threading.Thread(target=move_in_intersection, args=(constant.STOP,))
            x.start()
        if key == ord('g'):
            while True:
                c_key = cv2.waitKey(1)
                if c_key == ord('g'):
                    break
                time.sleep(1)
        if key == ord('q'):
            break
    else:
        cv2.imshow("result", frame)
    rawCapture.truncate(0)
cap.release()
cv2.destroyAllWindows()
