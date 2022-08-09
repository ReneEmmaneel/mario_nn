import sys
import time
import sched
import logging
import random
import re
import os

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

buttons = ["A", "B", "X", "Y", "up", "Down", "Left", "Right"]

def add_output_to_file(filename, id, buttons):
    """Buttons: boolean list, see list of buttons"""
    if not os.path.exists(filename):
        file = open(filename, 'w')
        file.write("id,A,B,X,Y,Up,Down,Left,Right\n")
        file.close()

    file = open(filename, 'a')
    file.write(str(id) + ',' + ','.join([str(button) for button in buttons]) + '\n')
    file.close()

class FileChangeHandler(FileSystemEventHandler):
    def __init__(self):
        pass

    def on_created(self, event):
        if event.is_directory:
            print(event)
        else:
            event_file_name = event.src_path.split('\\')

            if event_file_name[1].split('.')[1] == "csv":
                print(event)
            else:
                id = int(re.sub("[^0-9]", "", event_file_name[1]))
                output_file = event_file_name[0] + "/outputFile.csv"

                buttons = []
                num = int(id/10)
                for _ in range(8):
                    bit = num % 2
                    buttons.append(int(bit))
                    num = num / 2
                buttons.reverse()

                add_output_to_file(output_file, id, buttons)

    def on_modified(self, event):
        pass#print(event)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    path = 'data/'
    event_handler = FileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()