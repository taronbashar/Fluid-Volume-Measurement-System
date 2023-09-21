# from picamera.array import PiRGBArray
# from picamera import PiCamera
# import time

# def turn_on_lights():
#     pass

# def turn_off_lights():
#     pass

# def take_picture():

#     turn_on_lights()
    
#     camera = PiCamera()
#     rawCapture = PiRGBArray(camera)

#     # warmup camera
#     time.sleep(0.1)

#     camera.capture(rawCapture, format="bgr")
#     image = rawCapture.array
    
#     turn_off_lights()
    
#     return image