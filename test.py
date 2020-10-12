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
default_angle = 19.0
default_large_angle = 23.0
default_speed = 17.0
speed = 0.0
adjusted_speed = 0.0
angle = -1.15
moveAllowed=True

def forward(pwm, angle=0.0):
    if pwm >= 24.0:
        pwm = 23.0
    if pwm <= -24.0:
        pwm = -23.0
    if angle >= 25.0:
        angle = 24.0
    if angle <= 25.0:
        angle = 24.0
    serialHandler.sendMove(pwm, float(angle))
    
    
def stop(angle=0.0):
    serialHandler.sendBrake(float(angle))
    
    
def wait(ms):
    time.sleep(float(ms))
    
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    if moveAllowed:
        forward(speed + adjusted_speed, angle)
    else:
        stop(angle)
    image = frame.array
    # show the frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("z"): 
        break
    if key == ord("w"):
        speed += 1.0
    if key == ord("s"):
        speed -= 1.0
    if key == ord("y"):
        speed +=0.5
    if key == ord("h"):
        speed -= 0.5
    if key == ord("a"):
        if angle == 0.0 or angle == -default_large_angle:
            angle = -default_angle
            adjusted_speed = 1.5
        elif angle == default_angle or angle == default_large_angle:
            angle = 0.0
            adjusted_speed = 0.0
    if key == ord("d"):
        if angle == 0.0 or angle == default_large_angle:
            angle = default_angle
            adjusted_speed = 1.5
        elif angle == -default_angle or angle == -default_large_angle:
            angle = 0.0
            adjusted_speed = 0.0
    if key == ord("q"):
        if angle == 0.0 or angle == -default_angle:
            angle = - default_large_angle
            adjusted_speed = 1.5
        elif angle == default_angle or angle == default_large_angle:
            angle = 0.0
            adjusted_speed = 0.0
    if key == ord("e"):
        if angle == 0.0 or angle == default_angle:
            angle = default_large_angle
            adjusted_speed = 1.5
        elif angle == -default_angle or angle == -default_large_angle:
            angle = 0.0
            adjusted_speed = 0.0
    if key == ord("r"):
        oldSpeed = -speed
        stop(angle)
        time.sleep(0.5)
        forward(oldSpeed, angle)
        speed = oldSpeed
    if key == ord(" "):
        stop(0.0)
        if moveAllowed:
            moveAllowed = False
        else:
            moveAllowed = True
            if speed < 0:
                speed = -default_speed
            else:
                speed = default_speed
            
cv2.destroyAllWindows()    
stop(0.0)
    