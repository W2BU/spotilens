import keyboard
import pyautogui as pag
import numpy as np
import cv2 as cv
import tesserocr
import json

from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import namedtuple
from PIL import Image, ImageGrab

Point = namedtuple('Point', 'x y')


@dataclass
class Song:
    title: str = None
    artist: str = None
    album: str = None
    date_added: str = None
    duration: str = None


# Image regions to cut out: E sign, Music video sign
CONTROL_KEYS = '1 2 3'.split()
SCROLL_DELAY = 0.01
SCROLL_FACTOR = 0.25
IMG_SCALE_FACTOR = 3
DILATE_ITERATIONS = 10
DILATE_KERNEL = cv.getStructuringElement(cv.MORPH_RECT, (7, 3))
FILTER_THRESHOLD = 0.8

TES_LANGS = 'eng+jpn+rus'
TES_PATH = r'C:/Program Files/Tesseract-OCR/tessdata/.'
TES_THRESHOLD = 60
TESAPI = tesserocr.PyTessBaseAPI(
    path=TES_PATH,
    oem=tesserocr.OEM.DEFAULT,
    psm=tesserocr.PSM.SINGLE_LINE,
    lang=TES_LANGS,
)

# parts of image to cut
subimg_to_cut = []
# top left, album column beginning, bottom right points
playlist_frame_points = []


def load_cut_regions():
    folder_name = 'cut'
    dest_path = Path.cwd() / folder_name

    is_empty = not any(dest_path.iterdir())

    if (not dest_path.exists()) or is_empty:
        print(
            f'No {folder_name} folder found or folder is empty. Recognition results might be inaccurate and contain invalid characters.'
        )

    for file in dest_path.iterdir():
        path_str = str(file)
        if path_str.endswith('.png'):
            img = cv.imread(path_str, cv.IMREAD_GRAYSCALE)
            subimg_to_cut.append(img)


def capture_points():
    points = []
    for key in CONTROL_KEYS:
        while True:
            if keyboard.is_pressed(key):
                point = pag.position()
                print(f'{key} point {point} captured')
                points.append(point)
                break

    playlist_frame_points.extend(points)


def scan_window() -> list[tuple]:
    tl, br = playlist_frame_points[0], playlist_frame_points[2]
    screenshot_box = (*tl, *br)
    scroll_value = int(-abs(tl.y - br.y) * SCROLL_FACTOR)

    pag.sleep(1)
    pag.moveTo(*br)

    previous_frame = None
    songs = []
    while True:
        current_frame = ImageGrab.grab(bbox=screenshot_box)

        pag.sleep(SCROLL_DELAY)
        pag.moveTo(*br)
        pag.scroll(scroll_value)

        if current_frame == previous_frame:
            break

        current_img = preprocess_image(current_frame)
        extracted_songs = extract_songs_from_image(current_img)
        songs.extend(extracted_songs)

        previous_frame = current_frame

    return songs


def preprocess_image(img: Image):
    operations = [
        make_grayscale,
        cut_subimgs,
        binarize,
        enlarge,
    ]
    opencv_img = pil_to_opencv(img)
    for op in operations:
        opencv_img = op(opencv_img)

    return opencv_img


def make_grayscale(img: np.array):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def binarize(img: np.array):
    _, res = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    return res


def cut_subimgs(img: np.array):
    res = img.copy()
    for template in subimg_to_cut:
        w, h = template.shape[::-1]
        match_point = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
        loc = np.where(match_point >= FILTER_THRESHOLD)
        for x, y in zip(*loc[::-1]):
            res[y : (y + h), x : (x + w)] = 0
    return res


def enlarge(img: np.array):
    return cv.resize(src=img, dsize=None, fx=IMG_SCALE_FACTOR, fy=IMG_SCALE_FACTOR)


def get_text_bboxes(img: np.array) -> list:
    dilated = cv.dilate(img, DILATE_KERNEL, iterations=DILATE_ITERATIONS)
    contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    boxes = []
    for contour in contours:
        # x, y, w, h = cv.boundingRect(contour)
        # cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

        boxes.append(cv.boundingRect(contour))

    # preview(img)
    return boxes


def extract_songs_from_image(img: np.array) -> list[tuple]:

    # normalazing relative to upscaled frame coordinate system
    normalized_points = [
        Point(
            (x - playlist_frame_points[0].x) * IMG_SCALE_FACTOR,
            (y - playlist_frame_points[0].y) * IMG_SCALE_FACTOR,
        )
        for x, y in playlist_frame_points
    ]

    title_artist_part = img[
        normalized_points[0].y : normalized_points[2].y,
        normalized_points[0].x : normalized_points[1].x,
    ].copy()

    # bounding boxes of title and artist
    title_artist_bboxes = get_text_bboxes(title_artist_part)
    all_bboxes = get_text_bboxes(img)
    # bounding boxes of album, date added, duration fields
    other_bboxes = list(set(all_bboxes) - set(title_artist_bboxes))
    # matching title and artist boxes with corresponding fields boxes into single group
    groups = get_bbox_groups(title_artist_bboxes, other_bboxes)
    # extracting text in boxes from img
    songs_on_img = groups_to_text(img, groups)

    return songs_on_img


def groups_to_text(img: np.array, groups: list[tuple]) -> list[tuple]:
    songs_on_img = []
    for group in groups:
        data = []
        is_confident = True
        for bbox in group:
            x, y, w, h = bbox
            # cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            # preview(img)
            img_region = img[y : (y + h), x : (x + w)]
            TESAPI.SetImage(opencv_to_pil(img_region))

            # if TESAPI.MeanTextConf() < TES_THRESHOLD:
            #     is_confident = False

            extracted_str = TESAPI.GetUTF8Text().rstrip()
            data.append(extracted_str)

        if is_confident:
            songs_on_img.append(tuple(data))

    return songs_on_img


def get_bbox_groups(title_artist_bboxes: list[tuple], other_bboxes: list[tuple]):

    # split text boxes into title-artist pairs
    # making use of fact that gap between songs is bigger than any other
    # countours are iterated from bottom up, so reversing order for convinience

    bboxes = np.array(title_artist_bboxes[::-1])
    diff = np.diff(bboxes[:, 1])
    max_diff = np.max(diff)
    idx = np.flatnonzero(np.abs(diff - max_diff) < (max_diff * 0.2)) + 1
    bbox_groups = np.split(bboxes, idx)
    bbox_pairs = [group for group in bbox_groups if len(group) == 2]

    song_bboxes = []
    if other_bboxes:
        other_bboxes = np.array(other_bboxes)
        for pair in bbox_pairs:
            title_bbox, artist_bbox = pair
            # Y coordinate is more than Y of title and less than Y+H of artist
            mask = (other_bboxes[:, 1] > (title_bbox[1])) & (
                other_bboxes[:, 1] < (artist_bbox[1] + artist_bbox[3])
            )
            misc_bboxes = other_bboxes[mask]
            # sorting in left to right order on image
            misc_bboxes = misc_bboxes[np.argsort(misc_bboxes[:, 0])]

            if len(misc_bboxes) == 3:
                # album_box, date_added_box, duration_box = song_data
                song_bboxes.append((title_bbox, artist_bbox, *misc_bboxes))

    return song_bboxes


def pil_to_opencv(img: Image):
    return cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)


def opencv_to_pil(opencv_img: np.array):
    return Image.fromarray(opencv_img)


def filter_duplicates(songs: list[tuple]) -> list[tuple]:
    # dicts preserve order
    filtered = list(dict.fromkeys(songs))
    return filtered


def tuples_to_song_class(song_tuples: list[tuple]) -> list[Song]:
    songs = []
    for stuple in song_tuples:
        songs.append(Song(*stuple))
    return songs


def uniquify_filename(base, ext, dest_dir):
    n = 1
    filename = f'{base}_{str(n)}{ext}'
    dest_path = dest_dir / filename
    while dest_path.exists():
        n += 1
        filename = f'{base}_{str(n)}{ext}'
        dest_path = dest_dir / filename

    return dest_path


def songs_to_json(songs: list[Song]):
    base = 'playlist_json'
    ext = '.txt'
    path = uniquify_filename(base, ext, dest_dir=Path.cwd())
    path.touch()

    songs_dicts = [asdict(song) for song in songs]
    with path.open(mode='w', encoding='utf-8') as f:
        json.dump(songs_dicts, fp=f, ensure_ascii=False, indent=4)


def print_to_file(songs: list[Song]):
    base = 'playlist'
    ext = '.txt'
    path = uniquify_filename(base, ext, dest_dir=Path.cwd())
    path.touch()

    with path.open(mode='w', encoding='utf-8') as f:
        f.write(
            f'NUMBER OF SONGS: {len(songs)}\nSCAN DATE: {datetime.today().strftime('%Y-%m-%d')}\n'
        )

        for i, song in enumerate(songs):
            f.write(f'{i+1}. {song.artist} - {song.title}\n')


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


def draw_rects(img, bboxes):
    for bbox in bboxes:
        x, y, w, h = bbox
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

    preview(img)


def run():
    pag.hotkey('alt tab'.split())
    load_cut_regions()
    capture_points()
    song_tuples = scan_window()
    filtered = filter_duplicates(song_tuples)
    songs = tuples_to_song_class(filtered)
    print_to_file(songs)
    songs_to_json(songs)


if __name__ == '__main__':
    run()
