import RPi.GPIO as GPIO
import time

                    # toate ledurile se aprind doar daca a fost pornita masina( ceva un flag start)
blueSignal = 19     # culoarea albastra din fata, lucreaza intodeuna dupa start
leftSignal = 6      # se activeaza la aplicarea unui unghi mai mare de 19 grade la stinga
rightSignal = 5     # se activeaza la aplicarea unui unghi mai mare de 19 grade la dreapta
stopSignal = 26     # se activeaza cind viteza este intre -5 si 5 cm/sec
backSignal = 21     # se activeaza cind viteza ii mai mica de -5 cm/sec
nightSignal = 13    # se activeaza cind de la arduino vine noapte si stinge cind vine zi
                    # (lucreaza in permanenta in perioada 01 noiembrie - 31 martie) optional

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(nightSignal, GPIO.OUT)  # primul argument ii pinul!
print("LED off")
GPIO.output(nightSignal, GPIO.LOW)
time.sleep(1)
