import keyboard
import pyautogui as pag
import numpy as np
import cv2 as cv

from dataclasses import dataclass
from PIL import Image, ImageGrab, ImageOps


@dataclass
class FrameData:
    box: tuple = None
    img: np.array = None


FRAME_NAMES = ['E SIGN', 'MUSIC VIDEO SIGN', 'TRACKS AREA']
FRAMES = {name: FrameData() for name in FRAME_NAMES}
CONTROL_KEYS = '1 2'.split()
SCROLL_DELAY = 0.01
threshold = 0.8
screenshots = []


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
    pag.moveTo(*neutral)
    pag.drag(1, 1)

    pag.press('end')
    pag.sleep(1)

    end_frame = ImageGrab.grab(bbox=capture_box)

    pag.press('home')
    pag.sleep(1)

    previous_frame = None

    while True:
        current_frame = ImageGrab.grab(bbox=capture_box)
        pag.sleep(SCROLL_DELAY)
        pag.scroll(scroll_speed)

        if current_frame == previous_frame:
            break

        # screenshots.append(current_frame)
        process_image(current_frame)
        previous_frame = current_frame


def process_image(img: Image):
    operations = [
        make_grayscale,
        filter_signs,
        make_black_and_white,
    ]
    opencv_img = pil_to_opencv(img)
    for op in operations:
        opencv_img = op(opencv_img)


def make_grayscale(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def make_black_and_white(img):
    _, res = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
    return res


def filter_signs(img):
    res = img.copy()
    signs = ['E SIGN', 'MUSIC VIDEO SIGN']
    for sign in signs:
        template = FRAMES[sign].img
        w, h = template.shape[::-1]
        match_point = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
        loc = np.where(match_point >= threshold)
        for pt in zip(*loc[::-1]):
            # cv.rectangle(res, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
            res[pt[1] : (pt[1] + h), pt[0] : (pt[0] + w)] = 0
    return res


def pil_to_opencv(img: Image):
    return cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)


def opencv_to_pil(cv_img):
    return Image.fromarray()


def run():
    pag.hotkey('alt tab'.split())
    capture_points()
    take_screenshots()
    print(FRAMES)


if __name__ == '__main__':
    run()
