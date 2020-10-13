from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import numpy as np
import cv2
import threading
import math
import serial
# import fake_signals as si
import signals as si

from SerialHandler import SerialHandler
from SerialHandler import ReadThread
from SerialHandler import FileHandler

#serialHandler = SerialHandler.SerialHandler('/dev/ttyACM0')
#serialHandler.startReadThread()

serialHandler = SerialHandler()
th = ReadThread(threading.Thread, serial.Serial('/dev/ttyACM0',256000,timeout=1), FileHandler('historyFile.txt'), True)
th.start()
si.init_gpi()
 
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (800, 608)
camera.framerate = 20
rawCapture = PiRGBArray(camera, size=(800, 608))
default_angle = 19.0
default_large_angle = 23.0
default_speed = 17.0
speed = 0.08
adjusted_speed = 0.0
angle = 0.0
moveAllowed=True
si.startup()

speed_add = 0.1
SPEED_STEP_REGULAR = 0.5
SPEED_STEP_SOFT = 0.02
ANGLE_STEP_REGULAR = 10.0
ANGLE_STEP_SOFT = 5.0
MAX_SPEED = 1.5
MAX_ANGLE = 23.0
is_reverse = False


size = (800, 608)
result = cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc(*'MJPG'),11, size)

def forward(pwm):
    global angle
    if pwm >= MAX_SPEED:
        pwm = MAX_SPEED
    if pwm <= -MAX_SPEED:
        pwm = -MAX_SPEED
    if angle >= MAX_ANGLE:
        angle = MAX_ANGLE
    if angle <= -MAX_ANGLE:
        angle = -MAX_ANGLE
    
    if angle > 18:
        si.signaling_right()
    if angle < -18:
        si.signaling_left()
    serialHandler.sendMove(pwm, float(angle))
    
def adjust_speed():
    global speed
    global angle
    global adjusted_speed
    return None
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
    

def signaling_left():
    x_thread = threading.Thread(target=si.signaling_left, args=())
    x_thread.start()


def signaling_right():
    x_thread = threading.Thread(target=si.signaling_right, args=())
    x_thread.start()
    
def signaling_all():
    x_thread = threading.Thread(target=si.signaling_all, args=())
    x_thread.start()
    
def do_finish():
    result.release()
    cv2.destroyAllWindows()    
    stop(0.0)
    si.finish()
    
wait(1)
si.startup()
wait(4)

signaling_left()
wait(3)
signaling_right()
wait(3)
si.finish()
wait(3)
exit(1)

    
    
try:
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
#         print(speed, ' angle: ', angle)
        if moveAllowed:
            rps = th.getSpeed()
            if is_reverse:
                if rps < -46:
                    speed += 0.01
                elif rps >= -42:
                    speed -= 0.008  
            else:
                if rps > 46:
                    speed -= 0.01
                elif rps <= 42:
                    speed += 0.008
            forward(speed)
        else:
            stop(angle)
        image = frame.array
        # show the frame
        cv2.imshow("Frame", image)
        result.write(image)
        key = cv2.waitKey(1) & 0xFF

        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        # if the `q` key was pressed, break from the loop
        if key == ord("z"): 
            break
        if key == ord("c"): # recentre angle
            angle = 0.0
        if key == ord("t") or key == ord("T"):
            speed += SPEED_STEP_REGULAR
        if key == ord("g") or key == ord("G"):
            speed -= SPEED_STEP_REGULAR
        if key == ord("y"):
            speed += SPEED_STEP_SOFT
        if key == ord("h"):
            speed -= SPEED_STEP_SOFT
        if key == ord("a"):
            angle -= ANGLE_STEP_SOFT
            adjust_speed()
        if key == ord("d"):
            angle += ANGLE_STEP_SOFT
            adjust_speed()
        if key == ord("q"):
            angle -= ANGLE_STEP_REGULAR
            adjust_speed()
        if key == ord("e"):
            angle += ANGLE_STEP_REGULAR
            adjust_speed()
        if key == ord("r"):
            oldSpeed = -speed
            si.stop_lights_on()
            stop(angle)
            wait(0.2)
            forward(oldSpeed)
            si.stop_lights_off()
            speed = oldSpeed
            is_reverse = not is_reverse
        if key == ord(" "):
            if moveAllowed:
                moveAllowed = False
                si.back_mode_on()
            else:
                moveAllowed = True
                si.back_mode_off()
except Exception as ex:
    print(ex)
    do_finish()
    

do_finish()
th.join()
