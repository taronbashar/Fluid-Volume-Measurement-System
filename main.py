from fvms.button import is_button_pressed
# from fvms.picture import take_picture
from fvms.lcd import display
from fvms.fluid import FluidImage
import os
import pathlib
import math

def bestfit_pixelcount_area(x):
    vol = 0.003860099274175191 * x + 21.051393802148105
    if vol < 100:
        return 0
    else:
        return vol

def bestfit_convexhull_area(x):
    vol = 0.0029159002794959016 * x + 83.43005307603075
    if vol < 100:
        return 0
    else:
        return vol

def bestfit_nheights_area(x):
    if 0 < x < 1000:
        return 0
    return 0.0017057211183978333 * x + 169.8723945767969

def bestfit_max_height(x):
    if math.isclose(x, 0.0):
        return 0
    return 6.422810055686096 * x + 75.1362403836868

def bestfit_nheights1(x):
    b = [0.0, 1.09140213, 6.97770464, 5.98425934, -2.40233225, -4.84872598]
    intercept = b[0]
    slopes = b[1:]
    
    return intercept + sum([slopes[i] * x[i] for i in range(len(x))])

def bestfit_nheights2(x):
    b = [0.0, 3.65874126, -2.88553352, 14.03589521, 1.39855245, -2.95000514, -6.4046796, -3.79249717, 6.21007324, 0.78117608, -2.24910791]
    intercept = b[0]
    slopes = b[1:]
    
    return intercept + sum([slopes[i] * x[i] for i in range(len(x))])
    

METHODS = {
    # error = 198.32967711367849
    "pixelcount_area": {
        "bestfit": bestfit_pixelcount_area,
        "kwargs": {},
        "crop": (350, 1520, 4300, 1650),
        "threshold": 185
    },
    # error = 231.1090858508968
    "convexhull_area": {
        "bestfit": bestfit_convexhull_area,
        "kwargs": {},
        "crop": (350, 1520, 4300, 1650),
        "threshold": 230
    },
    # error = 571.0665060261141
    "nheights_area": {
        "bestfit": bestfit_nheights_area,
        "kwargs": {"n": 10},
        "crop": (250, 1480, 4260, 1680),
        "threshold": 250
    },
    # error = 667.181202378193
    "max_height": {
        "bestfit": bestfit_max_height,
        "kwargs": {},
        "crop": (250, 1480, 4260, 1680),
        "threshold": 210
    },
    # error = 113.06579482467554
    "nheights1": {
        "bestfit": bestfit_nheights1,
        "kwargs": {"n": 5},
        "crop": (250, 1480, 4260, 1680),
        "threshold": 240
    },
    # error = 27.818392109301566
    "nheights2": {
        "bestfit": bestfit_nheights2,
        "kwargs": {"n": 10},
        "crop": (250, 1480, 4260, 1680),
        "threshold": 180
    }
}


METHOD = "pixelcount_area"

CROP = METHODS[METHOD]["crop"]
THRESHOLD = METHODS[METHOD]["threshold"]
BESTFIT = METHODS[METHOD]["bestfit"]
KWARGS = METHODS[METHOD]["kwargs"]

PICTURE_NAME = "picture.jpg"
PICTURE_PATH = str(pathlib.Path(os.getcwd(), PICTURE_NAME))
CMD = f'libcamera-jpeg --nopreview --height 2200 --width 4500 -o {PICTURE_PATH}'

SHOW_PREVIEW = False

def take_picture():
    os.system(CMD)
    return PICTURE_PATH

def main():
    
    top_line = "Volume: None"
    bottom_line = "Press button."
    
    b_update_display = True
    
    def update_display():
        nonlocal b_update_display
        if b_update_display:
            to_display = f"{top_line}\n{bottom_line}"
            display(to_display)
            print(to_display + "\n")
            b_update_display = False
    
    while True:
        
        update_display()
            
        if is_button_pressed():
            
            bottom_line = "Measuring..."
            b_update_display = True
            update_display()
            b_update_display = True
            
            image = take_picture()
            
            if SHOW_PREVIEW:
                fluid = FluidImage(image).show().grayscale().crop(CROP).show().threshold(THRESHOLD).show()
            else:
                fluid = FluidImage(image).grayscale().crop(CROP).threshold(THRESHOLD)
                
            volume = fluid.predict(METHOD, BESTFIT, **KWARGS)
            
            top_line = f"Volume: {volume:.2f}uL"
            bottom_line = "Press button."
            

if __name__ == "__main__":
    main()