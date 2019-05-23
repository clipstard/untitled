# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import numpy as np
import cv2
import SerialHandler
import threading

serialHandler = SerialHandler.SerialHandler('/dev/ttyACM0')
serialHandler.startReadThread()
 
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 20
rawCapture = PiRGBArray(camera, size=(640, 480))
 

# allow the camera to warmup
time.sleep(0.1)

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
    

def forward(pwm, angle=0.0):
    serialHandler.sendMove(pwm, float(angle))
    
    
def stop(angle=0.0):
    serialHandler.sendBrake(float(angle))
    
    
def wait(ms):
    time.sleep(float(ms))
    
    
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 0, 255)
    return canny


def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([
        [(-140, height),
         (0 + int(width / 6), int(height) - int(height / 2)),
         (width - int(width / 6), int(height) - int(height / 2)),
         (width + 140, height)]
    ])
    vid = np.array([
        [(0, height),
         (width, height),
         (int(width - width/5), int(height * 1.2) - int(height /3)),
         (int(width / 5), int(height * 1.2) - int(height / 3))
         ]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    cv2.fillPoly(mask, vid, 0)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


default_angle = 19.0
default_large_angle = 23.0
default_speed = 17.0
speed = 0.0
angle = -1.5

##forward(17.0, 0.0)
##time.sleep(0.4)
##forward(14.5, 0.0)
##time.sleep(0.3)
moveAllowed=True
#forward(20.0, 23.0)
#time.sleep(7.0)
#stop()
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    if moveAllowed:
        forward(speed, angle)
    image = frame.array
    canny_image = canny(image)
    cropped_image = region_of_interest(canny_image)
    cv2.imshow("test", cropped_image)
    # show the frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("z"): 
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
    if key == ord("l"):
        parking_action()
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
