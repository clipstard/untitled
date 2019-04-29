import cv2
import math
import numpy as np


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    print('slope: ', slope, 'intercept: ', intercept)
    y1 = image.shape[0]
    y2 = int((y1 * 2 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    print('x1: ', x1, 'y1: ', y1, 'x2: ', x2, 'y2: ', y2)

    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            print('x1: ', x1, 'y1: ', y1, 'x2: ', x2, 'y2: ', y2)
            if x2 > x1 and y2 > y1:
                left_fit.append([x1, y1, x2, y2])
            elif x2 == x1 or y2 == y1:
                continue
            else:
                right_fit.append([x1, y1, x2, y2])

        # print(left_fit, 'right_fit', right_fit)

        print('left_fit', left_fit)
        avg_left = [0, 0, 0, 0]
        avg_right = [0, 0, 0, 0]
        count_left = 0
        count_right = 0

        for x1, y1, x2, y2 in left_fit:
            avg_left[0] += x1
            avg_left[1] += y1
            avg_left[2] += x2
            avg_left[3] += y2
            count_left += 1

        for x1, y1, x2, y2 in right_fit:
            avg_right[0] += x1
            avg_right[1] += y1
            avg_right[2] += x2
            avg_right[3] += y2
            count_right += 1

        avg_left[0] = avg_left[0] / count_left
        avg_left[1] = avg_left[1] / count_left
        avg_left[2] = avg_left[2] / count_left
        avg_left[3] = avg_left[3] / count_left

        avg_right[0] = avg_right[0] / count_right
        avg_right[1] = avg_right[1] / count_right
        avg_right[2] = avg_right[2] / count_right
        avg_right[3] = avg_right[3] / count_right

        print('avg_left', avg_left)
        print('avg_right', avg_right)

        left_line = np.polyfit((avg_left[0], avg_left[1]), (avg_left[2], avg_left[3]), 1)
        right_line = np.polyfit((avg_right[0], avg_right[1]), (avg_right[2], avg_right[3]), 1)

        print(right_line, 'right_line')
        left_total = make_coordinates(image, left_line)
        right_total = make_coordinates(image, right_line)

        return np.array([left_total, right_total])
    else:
        return None


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 45, 190)
    return canny


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    print(lines)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 20)
    return line_image


def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    print(width, height)
    polygons = np.array([
        [(width, height), (width, int(height - height / 2)), (width / 2, height / 10), (0, int(height - height / 2)),
         (0, height)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


image = cv2.imread('testhome.png')
lane_image = np.copy(image)
canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)
# cv2.imshow('asda', cropped_image)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 200, np.array([]), maxLineGap=5, minLineLength=20)

averaged_lines = average_slope_intercept(lane_image, lines)
line_image = display_lines(lane_image, lines)

combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
cv2.imshow("resultq", line_image)
cv2.waitKey(0)

# afisarea imaginii in axele x si y, pentru a determina virfurile la triunghiul de detectarea liniilor,
# pentru region_of_interest:

plt.imshow(canny(image))
plt.show()


# # cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('video2.mp4')
# while(cap.isOpened()):
#     _, frame = cap.read()
#     canny_image = canny(frame)
#     cropped_image = region_of_interest(canny_image)
#     lines = cv2.HoughLinesP(cropped_image, 10, np.pi / 180, 100, np.array([]), maxLineGap=5, minLineLength=40)
#     print(len(lines))
#     averaged_lines = average_slope_intercept(frame, lines)
#     if(len(averaged_lines)>1):
#         print("dfgdfgdf")
#         line_image = display_lines(frame, averaged_lines)
#         combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
#         cv2.imshow("result", combo_image)
#     # cv2.imshow("result1", cropped_image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
