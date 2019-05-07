# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import SerialHandler
import threading

serialHandler = SerialHandler.SerialHandler('/dev/ttyACM0')
serialHandler.startReadThread()

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (800, 600)
camera.framerate = 20
rawCapture = PiRGBArray(camera, size=(800, 600))

# allow the camera to warmup
time.sleep(0.1)
speed = 13.0
angle = 0.0
moveAllowed = True


def switch_move_allowed():
    global moveAllowed
    if moveAllowed:
        moveAllowed = False
    else:
        moveAllowed = True


def commit_action():
    forward(speed, 0.0)
    time.sleep(2.0)
    stop()
    switch_move_allowed()


def forward(pwm, an=0.0):
    serialHandler.sendMove(pwm, an)


def stop(an=0.0):
    serialHandler.sendBrake(an)


# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    if moveAllowed:
        forward(speed, angle)
    image = frame.array

    # show the frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    if key == ord("g"):
        switch_move_allowed()
        x = threading.Thread(target=commit_action, args=())
        x.start()
    if key == ord("a"):
        angle = 15.0
    if key == ord("d"):
        angle = -15.0
    if key == ord("w"):
        speed += 1.0
    if key == ord("s"):
        speed -= 1.0
    if key == ord(" "):
        stop(0.0)
        switch_move_allowed()
cv2.destroyAllWindows()
stop(0.0)
