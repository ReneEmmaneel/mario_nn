import sys
import time
import sched
import logging
import random
import win32gui, win32ui, win32con, win32api

from pyKey import pressKey, releaseKey, press, sendSequence, showKeys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

buttons = ["z", "x", "c", "v", "UP", "RIGHT", "DOWN", "LEFT"]

def release_all_buttons(emulator):
    win32gui.SetForegroundWindow(emulator)
    time.sleep(0.01)
    for button in buttons:
        releaseKey(button)

def press_buttons(emulator, event):
    print(emulator, event)
    win32gui.SetForegroundWindow(ezxcmulator)
    time.sleep(0.01)
    for button in buttons:
        if random.random() < 0.5:
            pressKey(button)
        else:
            releaseKey(button)

class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, emulator):
        self.emulator = emulator

    def on_created(self, event):
        press_buttons(self.emulator, event)

    def on_modified(self, event):
        print(event)

def find_all_windows(name):
    result = []
    def winEnumHandler(hwnd, ctx):
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd) == name:
            result.append(hwnd)
    win32gui.EnumWindows(winEnumHandler, None)
    return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    path = 'data/'
    emulator = find_all_windows("Super Mario World (USA) [SNES] - BizHawk")[0]
    event_handler = FileChangeHandler(emulator)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
