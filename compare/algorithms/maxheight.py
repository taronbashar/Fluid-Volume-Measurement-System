from PIL import Image
import numpy as np
import cv2
from .common import get_crop

# from skimage import color


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
    # gray_img = Image.fromarray(gray_arr)
    # gray_img.show()

    blur = cv2.blur(gray_arr, (3, 3))

    # #Apply mask to current image
    # BW, mask = segment_image(blur, threshold)
    ret, mask = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

    # #Convert mask array to image
    # im3 = Image.fromarray(mask)
    # im3.show()
    # Image.fromarray(mask).show()
    # img = np.zeros(mask.shape)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    heights = [cv2.boundingRect(c)[3] for c in contours]
    maxheight = max(heights) if heights else 0
    # pixels = 0
    # for cnt in contours:
    #     hull = cv2.convexHull(cnt)
    #     pixels += cv2.contourArea(hull)
    # cv2.drawContours(img, [cnt], -1, (255, 0, 0), 2)
    # print(pixels)
    # im = Image.fromarray(img)
    # im.show()

    # Determine no. of pixels and pixel area from mask
    # pixels = cv2.countNonZero(mask)

    return maxheight


def image_to_volume(image_path, bestfit=None, threshold=None):
    # Opens a image in RGB mode
    if threshold is None:
        raise ValueError("Pass in a threshold value")

    maxheight_pixels = image_to_independent(image_path, threshold=threshold)

    if bestfit is None:
        p2mm = 1 / 152
        width_pixels = (1 / p2mm) * 25.4

        triangle_area_pixels = (1 / 2) * maxheight_pixels * width_pixels

        pixel_area = p2mm * p2mm
        total_area = pixel_area * triangle_area_pixels
        vol = total_area * 50.8

        return vol
    else:
        return bestfit(maxheight_pixels)
