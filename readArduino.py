import serial
usbCom = serial.Serial('/dev/ttyUSB0', 9600)
while True:
    print(usbCom.readline())
