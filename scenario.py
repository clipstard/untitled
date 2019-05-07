# import the necessary packages
import time
import SerialHandler
import threading

serialHandler = SerialHandler.SerialHandler('/dev/ttyACM0')
serialHandler.startReadThread()

default_speed = 14.0
default_angle = 19.0
default_large_angle = 23.0
default_angle_speed = 16.0
default_large_angle_speed = 19.0
# initialize the camera and grab a reference to the raw camera capture

# allow the camera to warmup
time.sleep(0.1)


def forward(pwm, angle=0.0):
    serialHandler.sendMove(pwm, angle)


def stop(angle=0.0):
    serialHandler.sendBrake(angle)


def wait(ms):
    time.sleep(ms)


def make_angle_move(direction=True):
    if direction:
        forward(default_angle_speed, default_angle)
    else:
        forward(default_angle_speed, -default_angle)
    wait(2.0)


def make_large_angle_move(direction=True):
    if direction:
        forward(default_large_angle_speed, default_large_angle)
    else:
        forward(default_large_angle_speed, -default_large_angle)
    wait(3.0)


speed = 13.0
angle = 0.0

forward(default_speed, 0.0)
wait(1.5)
stop(0.0)
