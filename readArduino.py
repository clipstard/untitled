import serial
usbCom =serial.Serial('/dev/ttyUSB0',9600)
while 1:
    print(usbCom.readline())
