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
    # Image.fromarray(mask).show()
    # Image.fromarray(mask).show()
    img = np.zeros(mask.shape)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # area = 0
    # hulls = []
    # for cnt in contours:
    #     hull = cv2.convexHull(cnt)
    #     area += cv2.contourArea(hull)
    #     # hulls.append((hull, area))
    #     hulls.append(hull)
    hulls = [cv2.convexHull(c) for c in contours]
    cv2.drawContours(img, hulls, -1, (255, 0, 0), -1)
    # hulls.sort(key=lambda x: x[1])
    # hulls = [h[0] for h in hulls]
    # cv2.drawContours(img, [hulls[0]], -1, (255, 0, 0), -1)
    # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]

    # print(len(hulls))
    # print(area, cv2.countNonZero(img), cv2.contourArea(hulls[0]))
    # im = Image.fromarray(img)
    # im.show()

    area = cv2.countNonZero(img)

    # Determine no. of pixels and pixel area from mask
    # pixels = cv2.countNonZero(mask)

    return area


def image_to_volume(image_path, bestfit=None, threshold=None):
    # Opens a image in RGB mode
    if threshold is None:
        raise ValueError("Pass in a threshold value")

    hull_area = image_to_independent(image_path, threshold=threshold)

    if bestfit is None:
        p2mm = 1 / 152
        pixel_area = p2mm * p2mm
        total_area = pixel_area * hull_area
        vol = total_area * 50.8

        return vol
    else:
        return bestfit(hull_area)
