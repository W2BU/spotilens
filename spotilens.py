import keyboard
import pyautogui as pag
import numpy as np
import cv2 as cv
import pytesseract
import pprint

from pytesseract import Output
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from PIL import Image, ImageGrab


@dataclass
class FrameData:
    box: tuple = None
    img: np.array = None


# SIGN 1 - E(explicit)
# SIGN 2 - MUSIC VIDEO
SELECTION_ORDER = ['SIGN 1', 'SIGN 2', 'TRACKS AREA']
FRAMES = {name: FrameData() for name in SELECTION_ORDER}
CONTROL_KEYS = '1 2'.split()
SCROLL_DELAY = 0.01
SCROLL_FACTOR = 0.3
IMG_SCALE_FACTOR = 3
DILATE_ITERATIONS = 10  # pure magic 10
FILTER_THRESHOLD = 0.8
INDENT = 4

TES_LANGS = 'eng+jpn+rus'
TES_CFG = r'--psm 7 --oem 3 '

SONGS = []

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


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
    scroll_speed = int(-abs(tl.y - br.y) * SCROLL_FACTOR)

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
    ]
    opencv_img = pil_to_opencv(img)
    for op in operations:
        opencv_img = op(opencv_img)

    extracted_songs = extract_songs_from_image(opencv_img)
    print(extracted_songs)
    # print_to_file(extracted_songs)


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
        loc = np.where(match_point >= FILTER_THRESHOLD)
        for pt in zip(*loc[::-1]):
            res[pt[1] : (pt[1] + h), pt[0] : (pt[0] + w)] = 0
    return res


def upscale(img):
    return cv.resize(src=img, dsize=None, fx=IMG_SCALE_FACTOR, fy=IMG_SCALE_FACTOR)


def extract_songs_from_image(img):
    songs_on_img = []
    # horizontal line kernel
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 3))
    dilated = cv.dilate(img, kernel, iterations=DILATE_ITERATIONS)
    contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    rect = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        # rect.append((x, y, w, h))
        rect.append(cv.boundingRect(contour))
        # cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

    # split text countours into title-artist pairs
    # making use of fact that gap between songs is bigger than any other
    # countours are iterated from bottom up, so reversing order for convinience

    rect_a = np.array(rect[::-1])
    diff = np.diff(rect_a[:, 1])
    max_diff = np.max(diff)
    idx = np.flatnonzero(np.abs(diff - max_diff) < (max_diff * 0.2)) + 1
    groups = np.split(rect_a, idx)

    # print('\n\n')
    # print(f'GROUPS:\n{groups}\n')
    # print(f'IDX:\n{np.diff(idx)}\n')
    # print(f'GROUP OUT:')

    for group in groups:

        # accepring only tuples of 2 items: title and artist
        # filtered out items will appear on following screenshots

        if len(group) == 2:
            data = []
            for box in group:
                x, y, w, h = box
                x = x - INDENT
                y = y - INDENT
                w = w + INDENT
                y = y + INDENT
                img_region = img[y : (y + h), x : (x + w)]
                extracted_str = pytesseract.image_to_string(
                    img_region, lang=TES_LANGS, config=TES_CFG
                )
                data.append(extracted_str)
            songs_on_img.append(tuple(data))

    return songs_on_img


def print_to_file(data):
    filename = 'playlist.txt'
    dest_path = Path.cwd() / filename
    if not dest_path.exists():
        dest_path.touch()

    with dest_path.open(mode='w', encoding='utf-8') as f:
        for item in data:
            f.write(item)


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
    # cv.imshow('img', img)
    cv.waitKey(0)


def run():
    pag.hotkey('alt tab'.split())
    capture_points()
    take_screenshots()


if __name__ == '__main__':
    run()
