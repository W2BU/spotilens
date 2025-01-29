import keyboard
import pyautogui as pag
from dataclasses import dataclass
from PIL import Image, ImageGrab


@dataclass
class Frame:
    name: str = None
    box: tuple = None
    img: Image = None


FRAME_NAMES = ['E SIGN', 'MUSIC VIDEO SIGN', 'TRACKS AREA', ]
CAPTURED_FRAMES = ['E SIGN', 'MUSIC VIDEO SIGN']
FRAMES = [Frame(name=n) for n in FRAME_NAMES]
CONTROL_KEYS = '1 2'.split()


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

def take_screens():
    

if __name__ == '__main__':
    pag.press('alt tab'.split())
    capture_points()
    print(FRAMES)
