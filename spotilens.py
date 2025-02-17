import keyboard
import pyautogui as pag
import numpy as np
import cv2 as cv
import tesserocr

from datetime import datetime
from pathlib import Path
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
DILATE_KERNEL = cv.getStructuringElement(cv.MORPH_RECT, (5, 3))
FILTER_THRESHOLD = 0.8

TES_LANGS = 'eng+jpn+rus'
TES_PATH = r'C:/Program Files/Tesseract-OCR/tessdata/.'
tesapi = tesserocr.PyTessBaseAPI(
    path=TES_PATH,
    oem=tesserocr.OEM.DEFAULT,
    psm=tesserocr.PSM.SINGLE_LINE,
    lang=TES_LANGS,
)


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


def scan_window() -> list[tuple]:
    tl, br = FRAMES['TRACKS AREA'].box
    capture_box = (*tl, *br)
    neutral = ((tl.x + br.x) / 2, (tl.y + br.y) / 2)
    scroll_speed = int(-abs(tl.y - br.y) * SCROLL_FACTOR)

    pag.sleep(1)
    pag.moveTo(*neutral)

    previous_frame = None
    songs = []

    while True:
        current_frame = ImageGrab.grab(bbox=capture_box)
        pag.sleep(SCROLL_DELAY)
        pag.scroll(scroll_speed)

        if current_frame == previous_frame:
            break

        current_img = preprocess_image(current_frame)
        extracted_songs = extract_songs_from_image(current_img)
        songs.extend(extracted_songs)
        # print(extracted_songs)

        previous_frame = current_frame

    return songs


def preprocess_image(img: Image):
    operations = [
        make_grayscale,
        filter_signs,
        make_black_and_white,
        upscale,
    ]
    opencv_img = pil_to_opencv(img)
    for op in operations:
        opencv_img = op(opencv_img)

    return opencv_img


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


def get_bounding_boxes(img) -> list:
    dilated = cv.dilate(img, DILATE_KERNEL, iterations=DILATE_ITERATIONS)
    contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    boxes = []
    for contour in contours:
        # x, y, w, h = cv.boundingRect(contour)
        # rect.append((x, y, w, h))
        boxes.append(cv.boundingRect(contour))
        # cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

    return boxes


def extract_songs_from_image(img) -> list[tuple]:
    songs_on_img = []
    boxes = get_bounding_boxes(img)
    # horizontal line kernel

    # split text countours into title-artist pairs
    # making use of fact that gap between songs is bigger than any other
    # countours are iterated from bottom up, so reversing order for convinience

    rect_a = np.array(boxes[::-1])
    diff = np.diff(rect_a[:, 1])
    max_diff = np.max(diff)
    idx = np.flatnonzero(np.abs(diff - max_diff) < (max_diff * 0.2)) + 1
    groups = np.split(rect_a, idx)

    for group in groups:

        # accepring only tuples of 2 items: title and artist
        # filtered out items will appear on following screenshots

        if len(group) == 2:
            data = []
            for box in group:
                x, y, w, h = box
                img_region = img[y : (y + h), x : (x + w)]
                tesapi.SetImage(opencv_to_pil(img_region))
                extracted_str = tesapi.GetUTF8Text()
                data.append(extracted_str)
            songs_on_img.append(tuple(data))

    return songs_on_img


def print_to_file(data):
    filename = 'playlist.txt'
    dest_path = Path.cwd() / filename
    if not dest_path.exists():
        dest_path.touch()

    with dest_path.open(mode='w', encoding='utf-8') as f:
        f.write(
            f'NUMBER OF SONGS: {len(data)}\nSCAN DATE: {datetime.today().strftime('%Y-%m-%d')}\n'
        )

        for i, item in enumerate(data):
            title, artist = item
            f.write(f'{i}. {artist} - {title}')


def pil_to_opencv(img: Image):
    return cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)


def opencv_to_pil(opencv_img):
    return Image.fromarray(opencv_img)


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


def filter_duplicates(songs: list[tuple]) -> list[tuple]:
    # Python 3.7: dictionaries preserve order
    filtered = list(dict.fromkeys(songs))
    return filtered


def run():
    pag.hotkey('alt tab'.split())
    capture_points()
    songs = scan_window()
    filtered = filter_duplicates(songs)
    print_to_file(filtered)


if __name__ == '__main__':
    run()
