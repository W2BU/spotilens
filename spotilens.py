import keyboard
import pyautogui as pag
import numpy as np
import cv2 as cv
import tesserocr
import json

from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from collections import namedtuple
from statistics import median, mean
from PIL import Image, ImageGrab

Point = namedtuple('Point', 'x y')


@dataclass(frozen=True)
class Song:
    title: str = None
    artist: str = None
    album: str = field(default=None, hash=False, compare=False)
    date_added: str = field(default=None, hash=False, compare=False)
    duration: str = field(default=None, hash=False, compare=False)


# Image regions to cut out: E sign, Music video sign
CONTROL_KEYS = '1 2'.split()
SCROLL_DELAY = 0.01
SCROLL_FACTOR = 0.25
IMG_SCALE_FACTOR = 2
DILATE_ITERATIONS = 2
DILATE_KERNEL = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
FILTER_THRESHOLD = 0.8
N_PIXELS_THRESHOLD = 10 * IMG_SCALE_FACTOR
REPLACE_RULES = {'|': 'I', '{': '('}

TES_LANGS = 'eng+jpn+rus'
TES_PATH = r'C:/Program Files/Tesseract-OCR/tessdata/.'
TES_THRESHOLD = 40
TESAPI = tesserocr.PyTessBaseAPI(
    path=TES_PATH,
    oem=tesserocr.OEM.DEFAULT,
    psm=tesserocr.PSM.SINGLE_COLUMN,
    lang=TES_LANGS,
)

# parts of image to cut
subimg_to_cut = []
# top left, album column beginning, bottom right point of playlist window
playlist_frame_points = []

break_found = False


def set_dilation_kernel():
    global DILATE_KERNEL
    w, h = get_screen_resolution()
    kw, kh = int(0.015 * IMG_SCALE_FACTOR * w), int(0.0025 * IMG_SCALE_FACTOR * h)
    DILATE_KERNEL = cv.getStructuringElement(cv.MORPH_RECT, (kw, kh))


def get_screen_resolution():
    img = ImageGrab.grab()
    return img.size


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


def scan_window():
    tl, br = playlist_frame_points[0], playlist_frame_points[-1]
    screenshot_box = (*tl, *br)
    scroll_value = int(-abs(tl.y - br.y) * SCROLL_FACTOR)

    pag.sleep(1)
    pag.moveTo(*br)

    previous_frame = None
    songs = []
    global break_found
    while True:
        current_frame = ImageGrab.grab(bbox=screenshot_box)

        pag.sleep(SCROLL_DELAY)
        pag.moveTo(*br)
        pag.scroll(scroll_value)

        if (current_frame == previous_frame) or break_found:
            break

        current_img = preprocess_image(current_frame)
        extracted_songs = extract_songs_from_image(current_img)
        songs.extend(extracted_songs)

        previous_frame = current_frame

    songs = filter_duplicates(songs)
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


def make_grayscale(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def binarize(img):
    _, res = cv.threshold(img, 180, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return res


def cut_subimgs(img):
    res = img.copy()
    for template in subimg_to_cut:
        w, h = template.shape[::-1]
        match_point = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
        loc = np.where(match_point >= FILTER_THRESHOLD)
        for x, y in zip(*loc[::-1]):
            res[y : (y + h), x : (x + w)] = 0
    return res


def enlarge(img):
    return cv.resize(src=img, dsize=None, fx=IMG_SCALE_FACTOR, fy=IMG_SCALE_FACTOR)


def get_text_bboxes(img):
    dilated = cv.dilate(img, DILATE_KERNEL)
    contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    bboxes = []
    for contour in contours:
        bboxes.append(cv.boundingRect(contour))

    return bboxes


def extract_songs_from_image(img):
    row_bounds = find_text_rows(img)

    global break_found
    extracted_songs = []

    for u, l in row_bounds:
        current_row = img[u:l, :]
        bbox_group = get_text_bboxes(img=current_row)

        if len(bbox_group) >= 3:
            # sorting by x coordinate
            bbox_group = sorted(bbox_group, key=lambda x: x[0])
            # setting title field as first
            if bbox_group[0][1] > bbox_group[1][1]:
                bbox_group[1], bbox_group[0] = bbox_group[0], bbox_group[1]
        else:
            break_found = True
            print('exited by len')
            break

        song_data = text_from_bbox_group(img=current_row, bboxes=bbox_group)
        extracted_songs.append(song_data)

    return extracted_songs


def find_text_rows(img):
    dilated = cv.dilate(img, DILATE_KERNEL)
    hist = np.sum(dilated, axis=1) // 255

    h, w = img.shape[:2]
    upper_bounds = [
        y
        for y in range(h - 1)
        if hist[y] <= N_PIXELS_THRESHOLD and hist[y + 1] > N_PIXELS_THRESHOLD
    ]
    lower_bounds = [
        y
        for y in range(h - 1)
        if hist[y] > N_PIXELS_THRESHOLD and hist[y + 1] <= N_PIXELS_THRESHOLD
    ]

    # removing upaired bounds
    if upper_bounds[0] > lower_bounds[0]:
        del lower_bounds[0]

    if upper_bounds[-1] > lower_bounds[-1]:
        del upper_bounds[-1]

    # filtering by height
    heights = [l - u for u, l in zip(upper_bounds, lower_bounds)]
    median_height = median(heights)
    del_idx = []
    for i, height in enumerate(heights):
        if height < (0.3 * median_height):
            del_idx.append(i)

    for i in reversed(del_idx):
        del upper_bounds[i]
        del lower_bounds[i]

    # filtering by gap
    diffs = np.diff(np.array(upper_bounds), n=2)
    diff_mean = mean(diffs)

    for i, diff in enumerate(diffs[:-1]):
        if abs(diff) > (diff_mean + N_PIXELS_THRESHOLD):
            global break_found
            break_found = True
            print('exited by diff')
            del upper_bounds[i + 2 :]
            del lower_bounds[i + 2 :]
            break

    row_bounds = list(zip(upper_bounds, lower_bounds))
    return row_bounds


def text_from_bbox_group(img, bboxes):
    data_on_img = []
    for bbox in bboxes:
        x, y, w, h = bbox
        img_region = img[y : (y + h), x : (x + w)]
        TESAPI.SetImage(opencv_to_pil(img_region))
        extracted_str = TESAPI.GetUTF8Text().rstrip()
        extracted_str = adjust_str(extracted_str)
        data_on_img.append(extracted_str)

    return tuple_to_song(data_on_img)


def adjust_str(s):
    s = s.rstrip()
    for char, replacement in REPLACE_RULES.items():
        s = s.replace(char, replacement)
    s = ' '.join(s.split())
    return s


def pil_to_opencv(img: Image):
    return cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)


def opencv_to_pil(opencv_img: np.array):
    return Image.fromarray(opencv_img)


def filter_duplicates(songs: list[tuple]):
    # dicts preserve order
    filtered = list(dict.fromkeys(songs))
    return filtered


def tuple_to_song(song_data):
    song = Song()
    match len(song_data):
        case 4:
            song = Song(
                title=song_data[0],
                artist=song_data[1],
                album=song_data[2],
                duration=song_data[3],
            )
        case 5:
            song = Song(
                title=song_data[0],
                artist=song_data[1],
                album=song_data[2],
                date_added=song_data[3],
                duration=song_data[4],
            )
    return song


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
            f'NUMBER OF SONGS: {len(songs)}\nSCAN DATE: {datetime.today().strftime('%Y-%m-%dT%H:%M:%S')}\n'
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


def run():
    pag.hotkey('alt tab'.split())
    load_cut_regions()
    set_dilation_kernel()
    capture_points()
    songs = scan_window()
    print_to_file(songs)
    songs_to_json(songs)


if __name__ == '__main__':
    run()
