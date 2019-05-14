import SerialHandler
import threading
import serial
import time
import sys
import SaveEncoder


global serialHandler
# A=0.3386615784101943-0.5865790603779275j
# B=0.8997744537076356-0.26262039077918475j
# C=0.8997744537076356+0.26262039077918475j
# D=0.3386615784101943+0.5865790603779275j
# timer=4.0

A = 0.0+0.0j
B = 0.0+0.3333j
C = 0.0+0.66j
D = 0.0+1.0j
timer = 0.5


def testBezier(serialHandler):
    # Event test
    ev1 = threading.Event()
    serialHandler.readThread.addWaiter("SPLN", ev1,print)
    time.sleep(2.0)
    print("Start state0")
    # sent=serialHandler.sendBezierCurve(0.5051175777578528+0.5051175777578528j,0.7840863128094306+0.22614884270627506j,0.7840863128094306-0.22614884270627506j,0.5051175777578528-0.5051175777578528j,3.0,False)
    sent = serialHandler.sendBezierCurve(A, B, C, D, timer, False)
    if sent:
        isConfirmed = ev1.wait(timeout=1.0)
        if(isConfirmed):
            print("Response was received!")
        else:
            raise ConnectionError('Response', 'Response was not received!')

        ev1.clear()
        isConfirmed = ev1.wait(timer+0.5)
        if(isConfirmed):
            print("Terminated")
        else:
            raise ConnectionError('Response', 'Response was not received!')

    else:
        print("Sending problem")
    print("END_TEST")
    serialHandler.readThread.deleteWaiter("SPLN", ev1)
    time.sleep(0.0)


def testMOVEAndBrake(serialHandler):
    # Event test
    pwm = 0.33
    e = SaveEncoder.SaveEncoder("Encoder%.2f" % pwm+".csv")
    e.open()

    ev1 = threading.Event()
    ev2 = threading.Event()
    serialHandler.readThread.addWaiter("MCTL", ev1, e.save)
    serialHandler.readThread.addWaiter("BRAK", ev1, e.save)
    serialHandler.readThread.addWaiter("ENPB", ev2, e.save)
    sent = serialHandler.sendEncoderPublisher()
    if sent:
        isConfirmed = ev1.wait(timeout=1.0)
        if(isConfirmed):
            print("Deactivate encoder was confirmed!")
    else:
        raise ConnectionError('Response', 'Response was not received!')
    time.sleep(1.0)
    print("Start moving")
    sent = serialHandler.sendMove(pwm,0.0)
    if sent:
        isConfirmed = ev1.wait(timeout=3.0)
        if(isConfirmed):
            print("Moving was confirmed!")
        else:
            raise ConnectionError('Response', 'Response was not received!')

    else:
        print("Sending problem")
    time.sleep(3.0)
    ev1.clear()
    print("Start braking")
    sent = serialHandler.sendBrake(0.0)
    if sent:
        isConfirmed = ev1.wait(timeout=1.0)
        if(isConfirmed):
            print("Braking was confirmed!")
        else:
            raise ConnectionError('Response', 'Response was not received!')
    else:
        print("Sending problem")

    time.sleep(1.0)
    print("END_TEST")
    sent = serialHandler.sendEncoderPublisher(False)
    if sent:
        isConfirmed = ev1.wait(timeout=1.0)
        if(isConfirmed):
            print("Deactivate encoder was confirmed!")
    else:
        raise ConnectionError('Response', 'Response was not received!')

    serialHandler.readThread.deleteWaiter("BRAK", ev1)
    serialHandler.readThread.deleteWaiter("MCTL", ev1)
    serialHandler.readThread.deleteWaiter("ENPB", ev2)
    e.close()


def testBrake(serialHandler):
    # Event test
    ev1 = threading.Event()
    serialHandler.readThread.addWaiter("BRAK", ev1)
    time.sleep(2.0)

    print("Start moving")
    for i in range(0, 8):
        if i % 2 == 0:
            sent = serialHandler.sendBrake(-20.0)
        else:
            sent = serialHandler.sendBrake(20.0)
        if sent:
            isConfirmed = ev1.wait(timeout=1.0)
            if(isConfirmed):
                print("Response was received!")
            else:
                raise ConnectionError('Response', 'Response was not received!')
        else:
            print("Sending problem")
        time.sleep(1)
        ev1.clear()

    time.sleep(1)
    # sent=serialHandler.sendBrake(0.0)
    if sent:
        ev1.wait()
        print("Confirmed")
    serialHandler.readThread.deleteWaiter("MCTL", ev1)


def testPid(SerialHandler):
    ev1 = threading.Event()
    serialHandler.readThread.addWaiter("PIDA", ev1,print)
    sent = serialHandler.sendPidActivation(True)
    if sent:
        isConfirmed = ev1.wait(timeout=1.0)
        if(isConfirmed):
            print("Response was received!")
        else:
            raise ConnectionError('Response', 'Response was not received!')

    else:
        print("Sending problem")
    
    serialHandler.readThread.deleteWaiter("PIDA", ev1)


def testSafetyStop(SerialHandler):
    ev1 = threading.Event()
    serialHandler.readThread.addWaiter("SFBR", ev1)
    sent = serialHandler.sendSafetyStopActivation(True)
    if sent:
        isConfirmed = ev1.wait(timeout=1.0)
        if(isConfirmed):
            print("Response was received!")
        else:
            raise ConnectionError('Response', 'Response was not received!')

    else:
        print("Sending problem")
    serialHandler.readThread.deleteWaiter("SFBR", ev1)


def testDistancePub(SerialHandler):
    ev1 = threading.Event()
    serialHandler.readThread.addWaiter("DSPB", ev1)
    sent = serialHandler.sendDistanceSensorsPublisher(False)
    if sent:
        isConfirmed = ev1.wait(timeout=1.0)
        if(isConfirmed):
            print("Response was received!")
        else:
            raise ConnectionError('Response', 'Response was not received!')

    else:
        print("Sending problem")
    serialHandler.readThread.deleteWaiter("DSPB", ev1)


def testPidValue(SerialHandler):
    ev1 = threading.Event()
    serialHandler.readThread.addWaiter("PIDS", ev1)
    # (kp,ki,kd,tf)=(2.8184,7.0832,0.28036,0)
    (kp, ki, kd, tf) = (0.93143, 2.8, 0.0, 0.0001)
    sent = serialHandler.sendPidValue(kp, ki, kd, tf)
    if sent:
        isConfirmed = ev1.wait(timeout=1.0)
        if(isConfirmed):
            print("Response was received!")
        else:

            raise ConnectionError('Response', 'Response was not received!')

    else:
        print("Sending problem")
    serialHandler.readThread.deleteWaiter("PIDS", ev1)


def main():
    # Initiliazation
    global serialHandler
    serialHandler = SerialHandler.SerialHandler("COM27")
    serialHandler.startReadThread()
    try:
        testPid(serialHandler)
        testMOVEAndBrake(serialHandler)
        # testBezier(serialHandler)
    except Exception as e:
        print(e)
        pass

    # testBrake(serialHandler)
    time.sleep(0.5)
    serialHandler.close()


if __name__ == "__main__":
    main()
