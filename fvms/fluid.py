try:
    from PIL import Image

    PIL_INSTALLED = True
except:
    PIL_INSTALLED = False
import cv2
import numpy as np
import os
import typing


class FluidImage:
    crop_rect = (442, 1820, 4260, 1940)
    blur_kernel = (3, 3)
    slide_length = 76.2  # mm
    slide_width = 25.4  # mm
    slide_glass_length = 50.8  # mm
    slide_glass_width = 25.4  # mm
    fluid_density = 1.0043  # g/ml
    methods = [
        "max_height",
        "nheights",
        "nheights_area",
        "pixelcount_area",
        "convexhull_area",
    ]

    def __init__(self, from_item):
        if isinstance(from_item, (str, bytes, os.PathLike)):
            self.init_from_path(from_item)
        elif isinstance(from_item, FluidImage):
            self.init_from_other(from_item)
        elif isinstance(from_item, (list, np.ndarray)):
            self.init_from_array(from_item)
        else:
            raise ValueError(
                f"Invalid parameter for FluidImage. Expected Path or another FluidImage, got {type(from_item)}"
            )

    def init_from_array(self, arr: typing.Union[np.ndarray, list]):
        self.img_path = None
        self.img_orig = np.ndarray(arr)
        self.img = self.img_orig.copy()

    def init_from_other(self, other: "FluidImage"):
        self.img_path = other.img_path
        self.img_orig = other.img_orig
        self.img = other.img.copy()

    def init_from_path(self, img_path: typing.Union[str, bytes, os.PathLike]):
        self.img_path = img_path
        if PIL_INSTALLED:
            self.img_orig = Image.open(img_path)
        else:
            self.img_orig = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        self.img = np.array(self.img_orig, dtype=np.uint8)

    def copy(self):
        return FluidImage(self)

    def get_image(self):
        return self.img

    def reset(self):
        self.img = np.array(self.img_orig, dtype=np.uint8)
        return self

    def get_method(self, method):
        return {
            "max_height": self.max_height,
            "nheights": self.nheights,
            "nheights_area": self.nheights_area,
            "pixelcount_area": self.pixelcount_area,
            "convexhull_area": self.convexhull_area,
        }[method]

    def method(self, method, *args, **kwargs):
        return self.get_method(method)(*args, **kwargs)

    def predict(self, method, bestfit, *args, **kwargs):
        return bestfit(self.method(method, *args, **kwargs))

    @staticmethod
    def _crop(im, box):
        x, y, x2, y2 = box
        w = x2 - x
        h = y2 - y
        return im[y : y + h, x : x + w]

    def crop(self, crop_rect=None):
        if crop_rect:
            self.img = FluidImage._crop(self.img, crop_rect)
        else:
            self.img = FluidImage._crop(self.img, FluidImage.crop_rect)
        return self

    def grayscale(self):
        self.img = np.array(
            np.dot(self.img[..., :3], [0.299, 0.587, 0.114]), dtype=np.uint8
        )
        return self

    # set blur to False to not do blur before threshold. Set blur to (n1, n2) to set the blur kernel
    def threshold(self, threshold, blur=None):
        if blur is None:
            blur = FluidImage.blur_kernel

        if blur:
            self.img = cv2.blur(self.img, blur)
        self.img = cv2.threshold(self.img, threshold, 255, cv2.THRESH_BINARY)[1]
        return self

    # -------- Helpers (do not modify self) ----------- #

    def get_contours(self, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE):
        contours = cv2.findContours(self.img, mode, method)
        return contours[0] if len(contours) == 2 else contours[1]

    def get_convexhull(self, largest_only=False):
        hull_img = np.zeros(self.img.shape)
        contours = self.get_contours(method=cv2.CHAIN_APPROX_NONE)
        hulls = (
            [cv2.convexHull(c) for c in contours]
            if not largest_only
            else [cv2.convexHull(max(contours, key=cv2.contourArea))]
        )
        cv2.drawContours(hull_img, hulls, -1, (255, 0, 0), -1)
        return hull_img

    # ------ Extracting data from image (doesnt modify self) ----- #

    def max_height(self):
        contours = self.get_contours(method=cv2.CHAIN_APPROX_NONE)
        heights = [cv2.boundingRect(c)[3] for c in contours]
        return max(heights) if heights else 0

    # get n evenly spaced heights
    def nheights(self, n, space=False):
        contours = self.get_contours()
        if len(contours):
            c = max(contours, key=cv2.contourArea)
            img = np.zeros(self.img.shape)
            cv2.drawContours(img, [c], -1, (255, 0, 0), -1)

            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])
            y_bottom = extBot[1] + 1

            xr = extRight[0]
            xl = extLeft[0]
            delta = xr - xl
            spacing = delta // (n + 1)

            x_points = [xl + ((i + 1) * spacing) for i in range(n)]

            def first_nonzero(arr):
                mask = arr != 0
                return np.where(mask.any(axis=0), mask.argmax(axis=0), -1)

            y_points = [first_nonzero(img.T[x]).item() for x in x_points]

            heights = [y_bottom - y for y in y_points]

        else:
            heights = [0] * n
            spacing = 0

        return heights if not space else (heights, spacing)

    @staticmethod
    def polynomial_area(heights, spacing):
        x = [0]
        x += [((i + 1) * spacing) for i in range(len(heights))]
        x += [spacing * (len(heights) + 1)]
        y = [0]
        y += heights
        y += [0]

        correction = x[-1] * y[0] - y[-1] * x[0]
        main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
        return 0.5 * np.abs(main_area + correction)

    def nheights_area(self, n):
        (heights, spacing) = self.nheights(n, space=True)
        return FluidImage.polynomial_area(heights, spacing)

    # get nonzero pixel count
    def pixelcount_area(self):
        return cv2.countNonZero(self.img)

    # gets area of convex hull
    def convexhull_area(self, largest_only=False):
        return cv2.countNonZero(self.get_convexhull(largest_only=largest_only))

    # ------ For debugging ----- #

    def show(self, img=None):
        if not PIL_INSTALLED:
            raise ValueError("Cannot show image without PIL library")
        if img is None:
            Image.fromarray(self.img).show()
        else:
            Image.fromarray(img).show()
        return self
