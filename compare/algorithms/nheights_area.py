from PIL import Image
import numpy as np
import cv2
from .common import get_crop
from .nheights import image_to_independent as image_to_independent_nheights


def polygon_area(x, y):
    correction = x[-1] * y[0] - y[-1] * x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    return 0.5 * np.abs(main_area + correction)


def image_to_independent(image_path, threshold=None, nheights=1):
    if threshold is None:
        raise ValueError("Pass in a threshold value")

    (heights, spacing) = image_to_independent_nheights(
        image_path, threshold=threshold, nheights=nheights, space=True
    )

    x = [0]
    x += [((i + 1) * spacing) for i in range(len(heights))]
    x += [spacing * (len(heights) + 1)]
    y = [0]
    y += heights
    y += [0]

    return polygon_area(x, y)


def image_to_volume(image_path, bestfit=None, threshold=None, nheights=1):
    if threshold is None:
        raise ValueError("Pass in a threshold value")

    nheights_area = image_to_independent(
        image_path, threshold=threshold, nheights=nheights
    )

    if bestfit is None:
        p2mm = 1 / 152
        pixel_area = p2mm * p2mm
        total_area = pixel_area * nheights_area
        vol = total_area * 50.8

        return vol
    else:
        return bestfit(nheights_area)
