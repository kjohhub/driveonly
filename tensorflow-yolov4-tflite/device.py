from gpiozero import Button,LED,Buzzer


class Device:
    def __init__(self):
        led = LED(26)
        buzzer = Buzzer(6)
    def Alarm(on):
        if on:
            led.blink()
            buzzer.on()
        else:
            led.off()
            buzzer.off()
