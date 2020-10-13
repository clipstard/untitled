import RPi.GPIO as GPIO
import constant
import time

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
signaling_all_run = False



def do_nothing():
    return None


def wait(seconds):
    seconds = float(seconds)
    time.sleep(seconds)
    

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

def signaling_all():
    global signaling_all_run
    while (signaling_all_run):
        GPIO.output(constant.signals[constant.RIGHT_YELLOW], GPIO.HIGH)
        GPIO.output(constant.signals[constant.LEFT_YELLOW], GPIO.HIGH)
        wait(0.5)
        GPIO.output(constant.signals[constant.RIGHT_YELLOW], GPIO.LOW)
        GPIO.output(constant.signals[constant.LEFT_YELLOW], GPIO.LOW)
        wait(0.5)

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
    do_nothing()


def all_lights_on():
    GPIO.output(constant.signals[constant.LEFT_YELLOW], GPIO.HIGH)
    GPIO.output(constant.signals[constant.RIGHT_YELLOW], GPIO.HIGH)
    stop_lights_on()
    blue_light_on()
    night_light_on()


def all_lights_off():
    GPIO.output(constant.signals[constant.LEFT_YELLOW], GPIO.LOW)
    GPIO.output(constant.signals[constant.RIGHT_YELLOW], GPIO.LOW)
    stop_lights_off()
    blue_light_off()
    night_light_off()
    
def startup():
    blue_light_on()
    night_light_on()
    for i in range(0, 2):
        GPIO.output(constant.signals[constant.RIGHT_YELLOW], GPIO.HIGH)
        GPIO.output(constant.signals[constant.LEFT_YELLOW], GPIO.HIGH)
        wait(0.5)
        GPIO.output(constant.signals[constant.RIGHT_YELLOW], GPIO.LOW)
        GPIO.output(constant.signals[constant.LEFT_YELLOW], GPIO.LOW)
        wait(0.5)
        
def finish():
    blue_light_off()
    blue_light_off()
    for i in range(0, 2):
        GPIO.output(constant.signals[constant.RIGHT_YELLOW], GPIO.HIGH)
        GPIO.output(constant.signals[constant.LEFT_YELLOW], GPIO.HIGH)
        wait(0.33)
        GPIO.output(constant.signals[constant.RIGHT_YELLOW], GPIO.LOW)
        GPIO.output(constant.signals[constant.LEFT_YELLOW], GPIO.LOW)
        wait(0.33)
