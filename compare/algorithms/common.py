CROP_RECT = (442, 1820, 4260, 1940)


def set_crop(new_crop):
    global CROP_RECT
    CROP_RECT = new_crop


def get_crop():
    return CROP_RECT
