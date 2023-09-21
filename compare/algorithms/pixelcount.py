from PIL import Image
import numpy as np
import cv2
from .common import get_crop

# from skimage import color


# Image segmentation function
def segment_image(X, threshold):
    # Apply threshold operation
    BW = X > threshold

    # Create masked image
    masked_image = np.copy(X)
    masked_image[~BW] = 0

    return BW, masked_image


def image_to_independent(image_path, threshold=None):
    if threshold is None:
        raise ValueError("Pass in a threshold value")

    im = Image.open(image_path)

    # Image cropping
    im1 = im.crop(get_crop())

    # im1.show()
    # #Convert image to grayscale array
    gray_arr = np.array(im1.convert("L"))

    # # Shows the image in image viewer
    gray_img = Image.fromarray(gray_arr)
    # gray_img.show()

    # #Apply mask to current image
    BW, mask = segment_image(gray_arr, threshold)

    # #Convert mask array to image
    # im3 = Image.fromarray(mask)
    # im3.show()

    # Determine no. of pixels and pixel area from mask
    pixels = cv2.countNonZero(mask)

    return pixels


def image_to_volume(image_path, bestfit=None, threshold=None):
    # Opens a image in RGB mode
    if threshold is None:
        raise ValueError("Pass in a threshold value")

    bright_pixels = image_to_independent(image_path, threshold=threshold)

    if bestfit is None:
        p2mm = 1 / 152
        pixel_area = p2mm * p2mm
        total_area = pixel_area * bright_pixels
        vol = total_area * 50.8

        return vol
    else:
        return bestfit(bright_pixels)
