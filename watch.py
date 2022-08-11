import sys
import time
import sched
import logging
import random
import re
import os
import csv

import torch

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import train
from train import Module

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
    def __init__(self, model):
        self.model = model

    def on_created(self, event):
        if event.is_directory:
            output_file = os.path.join(event.src_path, 'outputFile.csv')
            if not os.path.exists(output_file):
                file = open(output_file, 'w')
                file.write("id,A,B,X,Y,Up,Down,Left,Right\n")
                file.close()
        else:
            event_file_name = event.src_path.split('\\')

            if not event_file_name[1].split('.')[1] == "csv":
                id = int(re.sub("[^0-9]", "", event_file_name[1]))
                output_file = event_file_name[0] + "/outputFile.csv"

                #Add previous 3 frames if possible
                input = {}
                if not os.path.isfile(output_file): #First frame
                    input["previous_points"] = []
                    input["screenshots"] = []
                else:
                    with open(output_file, newline='') as state_file_data:
                        reader = csv.DictReader(state_file_data, delimiter=',')
                        points = []
                        for row in reader:
                            if id - int(row["id"]) <= 30:
                                points.append(row)
                        input["previous_points"] = points
                    input["screenshots"] = [f"{event_file_name[0]}/screenshot_{point['id']}.png" for point in input["previous_points"]]

                #Add current frame
                input["screenshots"].append(f"{event_file_name[0]}/screenshot_{id}.png")

                next_inputs = train.get_next_input(self.model, input)

                #buttons = []
                #num = int(id/10)
                #for _ in range(8):
                #    bit = num % 2
                #    buttons.append(int(bit))
                #    num = num / 2
                #buttons.reverse()

                add_output_to_file(output_file, id, next_inputs)

    def on_modified(self, event):
        pass

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    path = 'data/'

    model_hparams = {"t": 4}
    optimizer_hparams={"lr": 0.1}
    model = Module(model_hparams, optimizer_hparams)
    state_dict = torch.load("models\\lightning_logs\\version_4\\checkpoints\\epoch=59-step=240.ckpt")["state_dict"]
    model.load_state_dict(state_dict)

    event_handler = FileChangeHandler(model)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()