
# from picamera import PiCamera
# from time import sleep
from gpiozero import Button

_button = Button(17)

def is_button_pressed():
    return _button.is_pressed

# button = Button(17)
# camera = PiCamera()
# camera.rotation = 270

# camera.start_preview()
# button.wait_for_press()
# camera.capture('/home/pi/image.jpg')
# camera.stop_preview()
