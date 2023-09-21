from PIL import Image
import numpy as np
import cv2
from .common import get_crop

# from skimage import color


def first_nonzero(arr):
    mask = arr != 0
    return np.where(mask.any(axis=0), mask.argmax(axis=0), -1)


def find_y(arr):
    s = len(arr)
    nonzero = first_nonzero(arr)
    return s - np.asscalar(nonzero)


def image_to_independent(image_path, threshold=None, nheights=1, space=False):
    if threshold is None:
        raise ValueError("Pass in a threshold value")

    im = Image.open(image_path)
    im1 = im.crop(get_crop())
    gray_arr = np.array(im1.convert("L"))

    blur = cv2.blur(gray_arr, (3, 3))
    mask = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)[1]

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    if len(contours):
        c = max(contours, key=cv2.contourArea)
        # if nheights == 1:
        #     # find max height
        #     r = cv2.boundingRect(c)
        #     return [r[3]]

        img = np.zeros(mask.shape)
        cv2.drawContours(img, [c], -1, (255, 0, 0), -1)

        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        # extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        y_bottom = extBot[1] + 1

        xr = extRight[0]
        xl = extLeft[0]
        delta = xr - xl
        spacing = delta // (nheights + 1)

        x_points = [xl + ((i + 1) * spacing) for i in range(nheights)]
        y_points = [first_nonzero(img.T[x]).item() for x in x_points]

        # r = cv2.boundingRect(c)
        # coord = zip(x_points, y_points)
        # img2 = Image.new(mode="RGB", size=(mask.shape[1], mask.shape[0]))
        # img2 = np.array(img2)
        # cv2.drawContours(img2, [c], -1, (255, 255, 255), -1)
        # for c in coord:
        #     cv2.circle(img2, c, radius=6, color=(255, 0, 0), thickness=-1)
        # cv2.rectangle(img2, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (0, 255, 0), 1)
        # Image.fromarray(img2).show()

        heights = [y_bottom - y for y in y_points]

    else:
        heights = [0] * nheights
        spacing = 1

    return heights if not space else (heights, spacing)


def image_to_volume(image_path, bestfit=None, threshold=None, nheights=1):
    # Opens a image in RGB mode
    if threshold is None:
        raise ValueError("Pass in a threshold value")

    heights_pixels = image_to_independent(
        image_path, threshold=threshold, nheights=nheights
    )

    if bestfit is None:
        maxheight_pixels = heights_pixels
        p2mm = 1 / 152
        width_pixels = (1 / p2mm) * 25.4

        triangle_area_pixels = (1 / 2) * maxheight_pixels * width_pixels

        pixel_area = p2mm * p2mm
        total_area = pixel_area * triangle_area_pixels
        vol = total_area * 50.8

        return vol
    else:
        return bestfit(heights_pixels)
