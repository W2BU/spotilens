import keyboard
import pyautogui as pag
import numpy as np
import cv2 as cv

from dataclasses import dataclass
from PIL import Image, ImageGrab


@dataclass
class FrameData:
    box: tuple = None
    img: np.array = None


# Features inside artist name
# SIGN 1 - E(explicit)
# SIGN 2 - MUSIC VIDEO
SELECTION_ORDER = ['SIGN 1', 'SIGN 2', 'TRACKS AREA']
FRAMES = {name: FrameData() for name in SELECTION_ORDER}
CONTROL_KEYS = '1 2'.split()
SCROLL_DELAY = 0.01
IMG_SCALE_FACTOR = 3
DILATE_ITERATIONS = 10  # pure magic 10
threshold = 0.8


def capture_points():
    for frame in FRAMES:
        points = []
        for key in CONTROL_KEYS:
            while True:
                if keyboard.is_pressed(key):
                    point = pag.position()
                    print(f'{key} point {point} for frame {frame} captured')
                    points.append(point)
                    break

        FRAMES[frame].box = tuple(points)
        img = pil_to_opencv(
            ImageGrab.grab(bbox=(*FRAMES[frame].box[0], *FRAMES[frame].box[1]))
        )
        FRAMES[frame].img = make_grayscale(img)


def take_screenshots():
    tl, br = FRAMES['TRACKS AREA'].box
    capture_box = (*tl, *br)
    neutral = ((tl.x + br.x) / 2, (tl.y + br.y) / 2)
    scroll_speed = int(-abs(tl.y - br.y) * 0.4)

    # focus window
    pag.sleep(1)
    pag.moveTo(*neutral)

    previous_frame = None

    while True:
        current_frame = ImageGrab.grab(bbox=capture_box)
        pag.sleep(SCROLL_DELAY)
        pag.scroll(scroll_speed)

        if current_frame == previous_frame:
            break

        process_image(current_frame)
        previous_frame = current_frame


def process_image(img: Image):
    operations = [
        make_grayscale,
        filter_signs,
        make_black_and_white,
        upscale,
        add_gaussian_blur,
    ]
    opencv_img = pil_to_opencv(img)
    for op in operations:
        opencv_img = op(opencv_img)

    extract_text(opencv_img)


def make_grayscale(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def make_black_and_white(img):
    _, res = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    return res


def filter_signs(img):
    res = img.copy()
    signs = ['SIGN 1', 'SIGN 2']
    for sign in signs:
        template = FRAMES[sign].img
        w, h = template.shape[::-1]
        match_point = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
        loc = np.where(match_point >= threshold)
        for pt in zip(*loc[::-1]):
            res[pt[1] : (pt[1] + h), pt[0] : (pt[0] + w)] = 0
    return res


def upscale(img):
    return cv.resize(src=img, dsize=None, fx=IMG_SCALE_FACTOR, fy=IMG_SCALE_FACTOR)


def add_gaussian_blur(img):
    return cv.GaussianBlur(img, (5, 5), 0)


def extract_text(img):
    res = img.copy()
    # horizontal line kernel
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 1))
    dilated = cv.dilate(img, kernel, iterations=DILATE_ITERATIONS)
    contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    rect = []
    for countour in contours:
        x, y, w, h = cv.boundingRect(countour)
        rect.append((x, y, w, h))
        # cv.rectangle(res, (x, y), (x + w, y + h), (255, 0, 255), 2)
    
    
    # split to groups title-artist
    # making use of fact that gap between title-artist pairs is equal and bigger than others
    # countours are iterated from bottom up, so reverse order for convenience
    rect_a = np.array(rect[::-1])
    diff = np.diff(rect_a[:, 1])
    idx = np.flatnonzero(diff - np.max(diff))
    groups = np.split(rect_a, idx)
    
    for group in groups:
        # accepting only valid groups of title-artist
        # groups that have partial entries will be processes on following iterations
        if len(group) == 2:
            

    # preview(res)


def pil_to_opencv(img: Image):
    return cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)


def opencv_to_pil(cv_img):
    return Image.fromarray()


def preview(img):
    cv.imshow(
        'img',
        cv.resize(
            src=img,
            dsize=None,
            fx=(1 / IMG_SCALE_FACTOR),
            fy=(1 / IMG_SCALE_FACTOR),
        ),
    )
    cv.waitKey(0)


def run():
    pag.hotkey('alt tab'.split())
    capture_points()
    take_screenshots()


if __name__ == '__main__':
    run()
