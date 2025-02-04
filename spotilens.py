import keyboard
import pyautogui as pag
from dataclasses import dataclass
from PIL import Image, ImageGrab, ImageOps


@dataclass
class Frame:
    name: str = None
    box: tuple = None
    img: Image = None


FRAME_NAMES = ['E SIGN', 'MUSIC VIDEO SIGN', 'TRACKS AREA']
CAPTURED_FRAMES = ['E SIGN', 'MUSIC VIDEO SIGN']
FRAMES = [Frame(name=n) for n in FRAME_NAMES]
CONTROL_KEYS = '1 2'.split()
SCROLL_DELAY = 0.01

screenshots = []


def capture_points():
    for frame in FRAMES:
        points = []
        for key in CONTROL_KEYS:
            while True:
                if keyboard.is_pressed(key):
                    point = pag.position()
                    print(f'{key} point {point} for frame {frame.name} captured')
                    points.append(point)
                    break

        frame.box = tuple(points)
        if frame.name in CAPTURED_FRAMES:
            first, second = frame.box
            img = ImageGrab.grab(bbox=(*first, *second))
            frame.img = img


def take_screenshots():
    tracks_frame, *_ = [f for f in FRAMES if f.name == 'TRACKS AREA']
    top_left, bot_right = tracks_frame.box
    capture_box = (*top_left, *bot_right)
    neutral = ((top_left.x + bot_right.x) / 2, (top_left.y + bot_right.y) / 2)
    scroll_speed = int(-abs(top_left.y - bot_right.y) * 0.4)

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

        screenshots.append(current_frame)
        previous_frame = current_frame


def process_image(img: Image):
    operations = [
        filter_signs,
        make_black_and_white,
        ImageOps.invert,
    ]
    for op in operations:
        img = op(img)


def make_black_and_white(img: Image):
    thresh = 200
    fn = lambda x: 255 if x > thresh else 0
    res = img.convert('L').point(fn, mode='1')
    return res


def filter_signs(img):
    pass


def run():
    pag.hotkey('alt tab'.split())
    capture_points()
    take_screenshots()
    print(FRAMES)


if __name__ == '__main__':
    run()
