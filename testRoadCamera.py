import cv2
import math
import numpy as np
import time
import constant
import threading

const_actions = [
    constant.RIGHT,
    constant.LEFT,
    constant.FORWARD,
    constant.RIGHT,
    constant.LEFT,
    constant.RIGHT,
    constant.LEFT,
    constant.FINISH
]


def wait(seconds):
    seconds = float(seconds)
    time.sleep(seconds)


def no_lines_action():
    global angle
    forward(0.0)
    wait(1)
    forward(-16, -angle)
    wait(0.75)


def custom_wait(seconds):
    now = time.time()
    while time.time() < now + seconds:
        pass


def after_stop_left(move_specific=None):
    forward(15.5)
    wait(1.6)
    forward(19.0, -22.0)
    wait(3.8)
    forward(0.0)
    do_nothing()


def after_stop_right(move_specific=None):
    forward(16)
    wait(1.1)
    forward(16, 22)
    wait(2.9)
    forward(0.0)
    do_nothing()


def parking_action():
    local_speed = 18
    forward(-local_speed + 1)
    wait(0.1)
    forward(-local_speed + 1, 22)
    wait(1.8)
    forward(-local_speed + 1, 0)
    wait(0.7)
    forward(-local_speed - 1, -22)
    wait(1.6)
    forward(-local_speed + 1, 0)
    wait(0.5)
    forward(0.0)
    wait(0.5)
    forward(local_speed - 1, 0)
    wait(2.1)
    stop()
    wait(1.5)
    forward(-local_speed + 1, 0)
    wait(1)
    stop()
    wait(0.5)
    forward(local_speed + 1, -22)
    wait(1.7)
    forward(local_speed - 2)
    wait(0.7)
    forward(local_speed - 1, 22)
    wait(1.6)
    forward(local_speed - 1)
    wait(0.5)
    stop()


def after_stop_forward():
    forward(17.5)
    wait(2)
    do_nothing()


def move_in_intersection(direction, delay=0.0, move_specific=None):
    global threads_off
    global listen_to_lines
    global base_speed
    global frame_count
    global action_index

    if direction == constant.LEFT and not threads_off:
        listen_to_lines = False
        x_thread = threading.Thread(target=signaling_left, args=())
        x_thread.start()
        after_stop_left(move_specific)
        listen_to_lines = True
    elif direction == constant.RIGHT and not threads_off:

        listen_to_lines = False
        x_thread = threading.Thread(target=signaling_right, args=())
        x_thread.start()
        after_stop_right(move_specific)
        listen_to_lines = True
    elif direction == constant.FORWARD and not threads_off:

        listen_to_lines = False
        forward(17.0)
        wait(2.50)
        listen_to_lines = True
    elif direction == constant.STOP and not threads_off:
        wait(delay)
        listen_to_lines = False
        forward(0.0)
        for i in range(0, 3):
            print("%d," % (i + 1))
            wait(0.5)
        print('go')
        listen_to_lines = True
        stop_lights_off()
        move_in_intersection(const_actions[action_index])
        action_index += 1
    elif not threads_off:
        wait(4.0)
    listen_to_lines = True
    frame_count = 0
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
    global height
    global horizontal_zone_from
    if array is None:
        return None
    aux = []
    for line in array:
        for item in line:
            x1, y1, x2, y2 = item
            abs_y = abs(y2 - y1)
            if abs_y == 0:
                abs_y = 1
            if abs(x1 - x2) / abs_y > 5 and y1 < height * horizontal_zone_from > y2:
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
    global height
    global horizontal_zone_from
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
        abs_y = abs(y2 - y1)

        if abs_y == 0:
            stop_fit.append([x1, y1, x2, y2])
        elif y1 > y2 and (abs(x2 - x1) / abs_y) < 5:
            left_fit.append((slope, intercept))
            left_lines.append([x1, y1, x2, y2])
        elif y1 < y2 and (abs(x2 - x1) / abs_y) < 5:
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


def display_average_lines(image, lines, interrupted):
    line_imagez = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            line = normalize_line(line)
            if line is not None:
                if isinstance(line, list):
                    x = line[0]
                    x1, y1, x2, y2 = x
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                else:
                    x1, y1, x2, y2 = line.reshape(4)
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                if y1 < y2 and interrupted[1] or y1 > y2 and interrupted[0]:
                    cv2.line(line_imagez, (x1, y1), (x2, y2), (0, 255, 0), 10)
                else:
                    cv2.line(line_imagez, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_imagez


def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([
        [(-140, height),
         (0 + int(width / 6), int(height) - int(height / 2)),
         (width - int(width / 6), int(height) - int(height / 2)),
         (width + 140, height)]
    ])
    # *******
    vid = np.array([
        [(50, height),
         (int(width / 5), height - int(height / 5.5)),
         (int(width - width / 5), height - int(height / 5.5)),
         (width - 50, height)
         ]
    ])
    # *******
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
        if current_lines[i] is None and last_lines[i] is not None:
            aux.append(normalize_line(convert_numpy_to_array(last_lines[i])))
        elif current_lines[i] is not None and last_lines[i] is None:
            aux.append(normalize_line(convert_numpy_to_array(current_lines[i])))
        else:
            aux.append(
                normalize_line(convert_numpy_to_array(make_average_coordinates([last_lines[i], current_lines[i]]))))
    return np.array(de_array(aux))


def down_angle(coefficient):
    global angle_coefficient
    coef = abs(coefficient)
    if coef < 0.6:
        value = 2
    elif 0.6 < coef < 1.0:
        value = coef * 5
    elif coef < 1.1:
        value = 11.25
    elif coef < 1.2:
        value = 12.5
    elif coef < 1.3:
        value = 14
    elif coef < 1.4:
        value = 16
    elif coef < 1.5:
        value = 18
    elif coef < 1.6:
        value = 20
    else:
        value = 23
    if value > 23:
        value = 23
    return -float(value)


def up_angle(coefficient):
    global angle_coefficient
    coef = abs(coefficient)
    if coef < 0.6:
        value = 2
    elif 0.6 < coef < 1.0:
        value = coef * 10
    elif coef < 1.1:
        value = 11.25
    elif coef < 1.2:
        value = 12.5
    elif coef < 1.3:
        value = 14
    elif coef < 1.4:
        value = 16
    elif coef < 1.5:
        value = 18
    elif coef < 1.6:
        value = 20
    else:
        value = 23
    if value > 23:
        value = 23
    return float(value)


def do_nothing():
    return None


def new_angle(lines):
    global height
    global width
    coefficients = [None, None]
    print(len(lines), '<= len lines')
    if len(lines) == 1:
        line = lines[0]
        x1, y1, x2, y2 = line
        if width - x1 > width / 2:
            coefficients[0] = x1
        else:
            coefficients[1] = (width - x1)
    if len(lines) == 2:
        left_line, right_line = lines
        if left_line is not None and right_line is not None:
            rx1, ry1, rx2, ry2 = right_line
            lx1, ly1, lx2, ly2 = left_line
            coefficients[0] = lx1
            coefficients[1] = width - rx1
        elif right_line is not None:
            rx1, ry1, rx2, ry2 = right_line
            coefficients[1] = (width - rx1)
        elif left_line is not None:
            lx1, ly1, lx2, ly2 = left_line
            coefficients[0] = lx1

    if len(lines) >= 3:
        left_line, right_line, stop_line = lines[0:3]
        if right_line is not None:
            rx1, ry1, rx2, ry2 = right_line
            coefficients[1] = width - rx1
        if left_line is not None:
            lx1, ly1, lx2, ly2 = left_line
            coefficients[0] = lx1

    return coefficients


def process_lines_spaces(spaces):
    left_space, right_space = spaces
    if left_space is None and right_space is None:
        return None
    if left_space is None:
        return right_space
    if right_space is None:
        return left_space
    return right_space


def convert_space_to_angle(space, calculated_coefficient):
    if space is None:
        space = 0
    if calculated_coefficient < 0:
        sign = -1
    else:
        sign = 1
    if -1 < calculated_coefficient < 1:
        space = space * 2.5
        calculated_coefficient = abs(calculated_coefficient * 9)
    else:
        calculated_coefficient = abs(calculated_coefficient * 10)
    if calculated_coefficient > 23:
        calculated_coefficient = 23
    aux = (calculated_coefficient * sign + float(space / 45))
    return validate_angle(aux)


def validate_angle(c_angle):
    if c_angle >= 22:
        return 23
    if c_angle <= -22.5:
        return -22
    return c_angle


def validate_increase(i_speed):
    global max_increase_speed
    if i_speed >= max_increase_speed - 1:
        return i_speed - 2
    if i_speed <= -max_increase_speed + 1:
        return i_speed + 1
    return i_speed


def prepare_speed(c_angle):
    global base_speed
    global increase_speed
    if float(-18) < c_angle > float(18):
        value = base_speed + validate_increase(increase_speed)
    else:
        value = base_speed + increase_speed
    return float(value)


def all_are_the_same_or_near(lines):
    if lines is None or len(lines) <= 1:
        return False
    reference = lines[0]

    if len(reference) < 2:
        return False
    l_ref, r_ref = reference[0:2]
    l_flag = False
    r_flag = False
    for i in range(1, len(lines)):
        l_line, r_line = lines[i][0:2]
        if (l_ref is None and l_line is not None) or (l_ref is not None and l_line is None):
            l_flag = True
        if (r_ref is None and r_line is not None) or (r_ref is not None and r_line is None):
            r_flag = True
        if l_ref is not None and l_line is not None and abs(l_ref[0] - l_line[0]) < 40:
            l_flag = True
        if r_ref is not None and r_line is not None and abs(r_ref[0] - r_line[0]) < 40:
            r_flag = True

    return l_flag and r_flag


def pop_first(items):
    if len(items) < 2:
        return []
    aux = []
    for i in range(1, len(items)):
        aux.append(items[i])
    return aux


def downgrade_speed():
    global base_speed
    base_speed -= 0.5


def car_started():
    global base_speed, speed_const
    wait(1)
    base_speed = speed_const - 1


def avg(item1, item2):
    return int((abs(item1) + abs(item2)) / 2)


def guess_space_direction(lines):
    if lines is not None:
        left_line, right_line = lines
        if left_line is None:
            return constant.RIGHT
        if right_line is None:
            return constant.LEFT
        lx1, ly1, lx2, ly2 = left_line
        rx1, ry1, rx2, ry2 = right_line
        if lx1 < rx1:
            return constant.RIGHT
        else:
            return constant.LEFT
    return constant.FORWARD


stop_stack = []
last_valid_angle = 0
close_thread = False
threads_off = False
max_increase_speed = 3.5
speed_const = 15
horizontal_zone_from = 0.8
horizontal_zone_to = 0.9
count = 0
last_lines = []
increase_speed = 0
base_speed = speed_const
backup_base_speed = speed_const
restore_speed = speed_const
speed = base_speed
angle = 0.0
decision = 0.0
action_index = 0
max = 0
min = 100
angle_coefficient = 10.5
is_brake = False
frame_count = 0
speed_accuracy_stack = []
listen_to_lines = True
urgent_break = False
space = 0
space_direction = constant.FORWARD
spacing = [None, None]


# Diff section
from picamera.array import PiRGBArray
from picamera import PiCamera
import SerialHandler
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)


def forward(speed, angle=0.0):
    serialHandler.sendMove(speed, angle)
    do_nothing()


def stop(angle=0.0):
    serialHandler.sendBrake(angle)
    do_nothing()


def count2():
    global close_thread
    wait(2)
    close_thread = True
    return True


def event_listener():
    global threads_off
    global close_thread
    import serial
    #    usb_com = serial.Serial('/dev/ttyUSB0', 9600)
    aux_thread = threading.Thread(target=count2, args=())
    aux_thread.start()
    while True:
        if threads_off or close_thread:
            break
        #        message = usb_com.readline()
        message = "123"
        if message == constant.IS_DAY:
            night_light_off()
            do_nothing()
        if message == constant.IS_NIGHT:
            night_light_on()
            do_nothing()
        if message == constant.OBJECT_IN_BACK:
            # object detected function
            do_nothing()
        if message == constant.OBJECT_IN_FRONT:
            # object detected function
            do_nothing()
    print('thread closed')


def signaling_left():
    for i in range(0, 2):
        GPIO.output(constant.signals[constant.LEFT_YELLOW], GPIO.HIGH)
        wait(0.5)
        GPIO.output(constant.signals[constant.LEFT_YELLOW], GPIO.LOW)
        wait(0.5)
    do_nothing()


def signaling_right():
    for i in range(0, 2):
        GPIO.output(constant.signals[constant.RIGHT_YELLOW], GPIO.HIGH)
        wait(0.5)
        GPIO.output(constant.signals[constant.RIGHT_YELLOW], GPIO.LOW)
        wait(0.5)
    do_nothing()


def stop_lights_on():
    GPIO.output(constant.signals[constant.STOP_LIGHT], GPIO.HIGH)
    do_nothing()


def stop_lights_off():
    GPIO.output(constant.signals[constant.STOP_LIGHT], GPIO.LOW)
    do_nothing()


def blue_light_on():
    GPIO.output(constant.signals[constant.BLUE_LIGHT], GPIO.HIGH)
    do_nothing()


def blue_light_off():
    GPIO.output(constant.signals[constant.BLUE_LIGHT], GPIO.LOW)
    do_nothing()


def night_light_on():
    GPIO.output(constant.signals[constant.NIGHT_LIGHT], GPIO.HIGH)
    do_nothing()


def night_light_off():
    GPIO.output(constant.signals[constant.NIGHT_LIGHT], GPIO.LOW)
    do_nothing()


def back_mode_on():
    GPIO.output(constant.signals[constant.BACK_LIGHT], GPIO.HIGH)
    do_nothing()


def back_mode_off():
    GPIO.output(constant.signals[constant.BACK_LIGHT], GPIO.LOW)
    do_nothing()


def init_gpi():
    GPIO.setup(constant.signals[constant.BACK_LIGHT], GPIO.OUT)
    GPIO.setup(constant.signals[constant.NIGHT_LIGHT], GPIO.OUT)
    GPIO.setup(constant.signals[constant.BLUE_LIGHT], GPIO.OUT)
    GPIO.setup(constant.signals[constant.STOP_LIGHT], GPIO.OUT)
    GPIO.setup(constant.signals[constant.RIGHT_YELLOW], GPIO.OUT)
    GPIO.setup(constant.signals[constant.LEFT_YELLOW], GPIO.OUT)


def test_program():
    signaling_left()
    forward(19.0, 0.0)
    wait(0.5)
    forward(19.0, 22)
    wait(2)
    forward(20)
    wait(1)
    forward(21.0, -22)
    wait(2)
    forward(20.0)
    wait(1)
    stop()


def test_parking():
    global listen_to_lines
    global urgent_break
    wait(6.5)
    listen_to_lines = False
    parking_action()
    urgent_break = True
    listen_to_lines = True


serialHandler = SerialHandler.SerialHandler('/dev/ttyACM0')
serialHandler.startReadThread()
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 24
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(0.1)
init_gpi()
try:
    blue_light_on()
    event_listener()
    for camera_frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        frame = camera_frame.array

        # end diff section
        if urgent_break:
            break
        frame_count += 1
        if listen_to_lines:
            if not is_brake:
                forward(speed, angle)
            else:
                forward(0.0, angle)
            canny_image = canny(frame)
            cropped_image = region_of_interest(canny_image)
            cv2.imshow("test", cropped_image)
            cv2.moveWindow('test', 0, 0)
            lines = cv2.HoughLinesP(cropped_image, 2, (np.pi / 180), 100, np.array([]), minLineLength=20, maxLineGap=10)
            height = frame.shape[0]
            width = frame.shape[1]
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
                cv2.moveWindow('result', 600, 0)
                count = 1
            else:
                if count < 2:
                    to_check_lines = last_lines
                    count += 1
                else:
                    to_check_lines = None
                    speed_accuracy_stack = []
                    line_image = display_average_lines(frame, to_check_lines, lines_interrupted)
                    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
                    cv2.imshow("result", combo_image)
            if to_check_lines is not None:
                if len(speed_accuracy_stack) < 6:
                    speed_accuracy_stack.append(to_check_lines[0:2])

                if len(speed_accuracy_stack) > 1 and not all_are_the_same_or_near(speed_accuracy_stack):
                    base_speed = backup_base_speed
                    speed_accuracy_stack = []
                    if 0 <= increase_speed > -max_increase_speed:
                        increase_speed -= 0.5
                        print('decrease')
                    elif increase_speed > 0:
                        increase_speed = 0
                        print('zero')
                    if increase_speed < -max_increase_speed / 2:
                        stop_lights_on()
                    else:
                        stop_lights_off()
                elif len(speed_accuracy_stack) >= 6:
                    if increase_speed < max_increase_speed:
                        if len(to_check_lines) >= 2:
                            if -5 < angle < 5:
                                do_nothing()
                            else:
                                increase_speed += 0.5
                                print('increase')
                    speed_accuracy_stack = pop_first(speed_accuracy_stack)
                calculated_angle = calculate_angle(convert_numpy_to_array(to_check_lines))
                if len(to_check_lines) == 3:
                    x1 = 0
                    x2 = 0
                    y1 = 0
                    y2 = 0
                    try:
                        line = to_check_lines[2]
                        if len(line) == 1 and isinstance(line[0], list):
                            line = line[0]
                        x1, y1, x2, y2 = line
                    except:
                        print('here')
                    if height * horizontal_zone_from < y2 <= height * horizontal_zone_to and \
                            height * horizontal_zone_from < y1 <= height * horizontal_zone_to and \
                            (x1 > width / 2 or abs(x1 - x2) > 150):
                        do_nothing()
                        stop_stack.append(True)
                    else:
                        stop_stack = []
                    if len(stop_stack) >= 3:
                        stop_stack = []
                        wait_time = 0
                        if space:
                            space_direction = guess_space_direction(to_check_lines[0:2])
                            if space_direction == constant.LEFT:
                                spacing = [space, None]
                            elif space_direction == constant.RIGHT:
                                spacing = [None, space]
                            else:
                                spacing = [None, None]
                        x = threading.Thread(target=move_in_intersection, args=(constant.STOP, wait_time, spacing,))
                        x.start()

                        if action_index >= len(const_actions):
                            while x.isAlive():
                                do_nothing()
                            break
                        do_nothing()
                if 1000 < calculated_angle < 15000:
                    angle = 0.0
                else:
                    if -2.5 < calculated_angle < 2.5:
                        test_var = new_angle(to_check_lines)
                        space = process_lines_spaces(test_var)
                        angle = validate_angle(convert_space_to_angle(space, calculated_angle))
                        print(calculated_angle, angle, space, speed, 'ass')
                speed = prepare_speed(angle)
                last_valid_angle = angle
                if -5.0 < angle < 5.0:
                    speed = base_speed
                if speed < -5.0:
                    back_mode_on()
                else:
                    back_mode_off()

            else:
                angle = last_valid_angle
                speed = prepare_speed(angle)
                no_lines_action()
                cv2.imshow("result", frame)
                c_key = cv2.waitKey(1)
                if c_key == ord('q'):
                    break
            key = cv2.waitKey(1)
            if key == ord('p'):
                x = threading.Thread(target=move_in_intersection, args=(constant.STOP,))
                x.start()
            if key == ord('q'):
                break
        else:
            cv2.imshow("result", frame)
            c_key = cv2.waitKey(1)
            if c_key == ord('q'):
                break
                # another diff section
        rawCapture.truncate(0)
    cv2.destroyAllWindows()
    stop()
except Exception as ex:
    print(ex)
    stop()
    cv2.destroyAllWindows()
blue_light_off()
stop()
threads_off = True
