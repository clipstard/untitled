from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import numpy as np
import cv2
import SerialHandler
import threading
import math

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
speed = 10.0
adjusted_speed = 0.0
angle = 0.0
moveAllowed=True

def forward(pwm):
    global angle
    if pwm >= 23.0:
        pwm = 23.0
    if pwm <= -23.0:
        pwm = -23.0
    if angle >= 23.0:
        angle = 23.0
    if angle <= -23.0:
        angle = -23.0
    serialHandler.sendMove(pwm, float(angle))
    
def adjust_speed():
    global speed
    global angle
    global adjusted_speed
    if speed >= 23.0:
        speed = 23.0
    if speed <= -23.0:
        speed = -23.0
    # + 10% to speed
    a = abs(angle)
    if a >= 11.0:
        a = a * 0.08
    else:
        a = a * 0.11
    adjusted_speed = a

def stop(angle=0.0):
    serialHandler.sendBrake(float(angle))
    
stop()
def wait(ms):
    time.sleep(float(ms))
    
try:
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        print(speed, ' angle: ', angle)
        if moveAllowed:
            forward(speed + adjusted_speed)
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
        if key == ord("c"): # recentre angle
            angle = 0.0
        if key == ord("w"):
            speed += 1.0
        if key == ord("s"):
            speed -= 1.0
        if key == ord("y"):
            speed +=0.5
        if key == ord("h"):
            speed -= 0.5
        if key == ord("a"):
            angle -= 1.0
            adjust_speed()
        if key == ord("d"):
            angle += 1.0
            adjust_speed()
        if key == ord("q"):
            angle -= 5.0
            adjust_speed()
        if key == ord("e"):
            angle += 5.0
            adjust_speed()
        if key == ord("1"):
            angle -= 10.0
            adjust_speed()
        if key == ord("3"):
            angle += 10.0
            adjust_speed()
        if key == ord("r"):
            oldSpeed = -speed
            stop(angle)
            wait(0.2)
            forward(oldSpeed)
            speed = oldSpeed
        if key == ord(" "):
            if moveAllowed:
                moveAllowed = False
            else:
                moveAllowed = True
except Exception as ex:
    print(ex)
    cv2.destroyAllWindows()    
    stop(0.0)        
cv2.destroyAllWindows()    
stop(0.0) 
