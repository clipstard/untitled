import cv2
import math
import numpy as np
import time
import constant
import threading
from picamera.array import PiRGBArray
from picamera import PiCamera
import SerialHandler


serialHandler = SerialHandler.SerialHandler('/dev/ttyACM0')
serialHandler.startReadThread()
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 20
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


def forward(speed, angle=0.0):
    serialHandler.sendMove(speed, angle)


def stop(angle=0.0):
    serialHandler.sendBrake(angle)
    do_nothing()


def after_stop_left():
    global speed
    forward(speed, -1.0)
    wait(0.7)
    forward(21.0, -23.0)
    wait(3.3)
    do_nothing()


def after_stop_right():
    global speed
    forward(speed, -1.0)
    wait(0.2)
    forward(18.0, 23.0)
    wait(3.0)
    do_nothing()
    
    
def parking_action():
    forward(-19, 22)
    wait(1.5)
    forward(-17, 0)
    wait(0.8)
    forward(-21, -22)
    wait(1.7)
    forward(-19, 0)
    wait(1.1)
    stop()
    wait(0.5)
    forward(18, 0)
    wait(2)
    stop()


def after_stop_forward():
    do_nothing()


def move_in_intersection(direction):
    global listen_to_lines
    global base_speed
    global frame_count
    listen_to_lines = False
    print(direction)
    if direction == constant.LEFT:
        after_stop_left()
        listen_to_lines = True
    elif direction == constant.RIGHT:
        after_stop_right()
        listen_to_lines = True
    elif direction == constant.FORWARD:
        wait(1.0)
        listen_to_lines = True
    elif direction == constant.STOP:
        stop()
        for i in range(0, 3):
            print("%d," % (i+1))
            wait(0.5)
        print('go')
        listen_to_lines = True
    else:
        wait(4.0)
        listen_to_lines = True
    base_speed = 17.5
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
    global horizontal_zone_to
    if array is None:
        return None
    aux = []
    for line in array:
        for item in line:
            x1, y1, x2, y2 = item
            abs_y = abs(y2 - y1)
            if abs_y == 0:
                continue
            if abs(x1 - x2) / abs_y > 5 and x1 < height * horizontal_zone_from > x2:
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
        elif y1 > y2 and (abs(x2-x1)/abs_y) < 5:
            left_fit.append((slope, intercept))
            left_lines.append([x1, y1, x2, y2])
        elif y1 < y2 and (abs(x2-x1)/abs_y) < 5:
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
        [(-155, height),
         (0 + int(width / 4), height - int(height / 3)),
         (width - int(width / 4), height - int(height / 3)),
         (width + 155, height)]
    ])
    vid = np.array([
        [(0, height),
         (width, height),
         (int(width / 2), height - int(height / 3))]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
#    cv2.fillPoly(mask, vid, 0)
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
            aux.append(normalize_line(convert_numpy_to_array(make_average_coordinates([last_lines[i], current_lines[i]]))))
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
            if rx1 < width:
                coefficients[1] = width - rx1
            if lx1 > 0:
                coefficients[0] = lx1
        elif right_line is not None:
            rx1, ry1, rx2, ry2 = right_line
            coefficients[1] = (height - rx1)
        elif left_line is not None:
            lx1, ly1, lx2, ly2 = left_line
            coefficients[0] = lx1

    if len(lines) == 3:
        left_line, right_line, stop_line = lines
        if right_line is not None:
            rx1, ry1, rx2, ry2 = right_line
            if rx1 < width:
                coefficients[1] = width - rx1
        if left_line is not None:
            lx1, ly1, lx2, ly2 = left_line
            if lx1 > 0:
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
    if right_space > left_space:
        return right_space
    else:
        return left_space


def convert_space_to_angle(space, calculated_coefficient):
    if space is None:
        space = 0
    space = space * 2.75
    if calculated_coefficient < 0:
        sign = -1
    else:
        sign = 1
    calculated_coefficient = abs(calculated_coefficient * 11)
    if calculated_coefficient > 23:
        calculated_coefficient = 23
    aux = (calculated_coefficient * sign) + float(space / 100)
    if aux > 23:
        return 23
    if aux < -23:
        return -23
    return aux


def validate_angle(c_angle):
    if c_angle >= 22:
        return 23
    if c_angle <= -22:
        return - 23
    return c_angle


def prepare_speed(c_angle):
    global base_speed
    global increase_speed
    if c_angle < -17:
        value = base_speed * 0.225 + base_speed + increase_speed
    elif c_angle < -13:
        value = base_speed * 0.185 + base_speed + increase_speed
    elif c_angle < -10:
        value = base_speed * 0.135 + base_speed + increase_speed
    elif c_angle < -8:
        value = base_speed
    elif c_angle > 18:
        value = base_speed * 0.165 + base_speed + increase_speed
    elif c_angle > 15:
        value = base_speed * 0.145 + base_speed + increase_speed
    elif c_angle > 12:
        value = base_speed * 0.12 + base_speed + increase_speed
    else:
        value = base_speed + increase_speed
    return float(value)


def all_are_the_same_or_near(lines):
    if lines is None or len(lines) <= 1:
        return False
    reference = lines[0]
    l_ref, r_ref = reference[0:2]
    l_flag = False
    r_flag = False
    for i in range(1, len(lines)):
        l_line, r_line = lines[i][0:2]
        if (l_ref is None and l_line is not None) or (r_ref is None and r_line is not None) or\
                (l_ref is not None and l_line is None) or (r_ref is not None and r_line is None):
            return False
        if l_ref is not None and l_line is not None and abs(l_ref[0] - l_line[0]) < 20:
            l_flag = True
        if r_ref is not None and r_line is not None and abs(r_ref[0] - r_line[0]) < 20:
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
    base_speed -=0.5
    
    
def car_started():
    global base_speed, speed_const
    wait(1)
    base_speed = speed_const - 1


speed_const = 17.5
horizontal_zone_from = 0.75
horizontal_zone_to = 0.95
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
try:
    for camera_frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        base_speed = backup_base_speed
        frame = camera_frame.array
        frame_count += 1
        if frame_count < 2:
            x = threading.Thread(target=car_started, args=())
            x.start()
        if not is_brake:
            forward(speed, angle)
        else:
            forward(0.0, angle)
#            stop(angle)
            print(angle, 'angle')
        if listen_to_lines:
            canny_image = canny(frame)
            cropped_image = region_of_interest(canny_image)
            downgrade_speed()
            cv2.imshow("test", cropped_image)
            lines = cv2.HoughLinesP(cropped_image, 2, (np.pi / 180), 100, np.array([]), minLineLength=20, maxLineGap=10)
            height = frame.shape[1]
            width = frame.shape[0]
            lines = reduce_invalid(lines, height, width)
            lines = reduce_horizontals2(lines)
            to_check_lines = None
            averaged_lines, lines_interrupted = average_slope_intercept(cropped_image, lines)
            downgrade_speed()
            if averaged_lines is not None:
                to_check_lines = make_average_lines(last_lines, averaged_lines)
                last_lines = averaged_lines
                line_image = display_average_lines(frame, to_check_lines, lines_interrupted)
                
                combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
                cv2.imshow("result", combo_image)
                count = 0
                downgrade_speed()
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
                downgrade_speed()
                if len(speed_accuracy_stack) < 5:
                    increase_speed = 0
                    speed_accuracy_stack.append(to_check_lines)
                if len(speed_accuracy_stack) > 1 and not all_are_the_same_or_near(speed_accuracy_stack):
                    base_speed = backup_base_speed
                    speed_accuracy_stack = []
                elif len(speed_accuracy_stack) >= 5:

                    if increase_speed < 3:
                        increase_speed += 0.5
                    speed_accuracy_stack = pop_first(speed_accuracy_stack)
                all_are_the_same_or_near(speed_accuracy_stack)
                calculated_angle = calculate_angle(convert_numpy_to_array(to_check_lines))
                if len(to_check_lines) == 3:
                    line = to_check_lines[2]
                    if len(line) == 1 and isinstance(line[0], list):
                        line = line[0]
                    x1, y1, x2, y2 = line
                    if height * horizontal_zone_to < y2 >= height * horizontal_zone_from and \
                            height * horizontal_zone_to < y1 >= height * horizontal_zone_from:
                        x = threading.Thread(target=move_in_intersection, args=(constant.STOP,))
                        x.start()
                        y = threading.Thread(target=move_in_intersection, args=(const_actions[action_index],))
                        y.start()
                        action_index += 1
                        if action_index == len(const_actions):
                            break
                if -0.1 < calculated_angle < 0.1:
                    angle = 0.0
                else:
                    if -2.5 < calculated_angle < 2.5:
                        test_var = new_angle(to_check_lines)
                        space = process_lines_spaces(test_var)
                        angle = validate_angle(convert_space_to_angle(space, calculated_angle))
                        print(speed, angle, space, calculated_angle)
                speed = prepare_speed(angle)

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
            cv2.imshow("result", frame)
        else:
            cv2.imshow("result", frame)
            c_key = cv2.waitKey(1)
            if c_key == ord('g'):
                break
        rawCapture.truncate(0)
    cv2.destroyAllWindows()
    stop()
except Exception as ex:
    print(ex)
    stop()
    cv2.destroyAllWindows()

