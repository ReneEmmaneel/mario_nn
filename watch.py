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
    def __init__(self, model_checkpoint_folder):
        self.model_checkpoint_folder = model_checkpoint_folder
        self.iter = 0
        self.load_model()
        self.dir_name = None
        self.prev_id = None

    def load_model(self):
        self.model_path, self.model = load_latest_model(self.model_checkpoint_folder)
        self.iter += 1
        print(f'Iteration {self.iter}\tLoaded model {self.model_path}\Deterministic: {self.iter%2==0}')

    def on_created(self, event):
        if event.is_directory:
            output_file = os.path.join(event.src_path, 'outputFile.csv')
            if not os.path.exists(output_file):
                file = open(output_file, 'w')
                file.write("id,A,B,X,Y,Up,Down,Left,Right\n")
                file.close()
            self.load_model()
        else:
            event_file_name = event.src_path.split('\\')

            if not event_file_name[1].split('.')[1] == "csv":
                id = int(re.sub("[^0-9]", "", event_file_name[1]))

                #Sometimes FIleSystemWatcher triggers twice, so check if new id is higher than previous id,
                #or if it is a new directory (in that case, reset the previous id)
                new_dir = False
                if not event_file_name[0] == self.dir_name:
                    new_dir = True
                    self.prev_id = 0
                self.dir_name = event_file_name[0]

                if id <= self.prev_id:
                    return #ignore false screenshot
                else:
                    self.prev_id = id


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

                next_inputs = train.get_next_input(self.model, input, deterministic=self.iter%2==0)

                add_output_to_file(output_file, id, next_inputs)

    def on_modified(self, event):
        pass

def load_latest_model(folder_path):
    ckpts = []
    for file in os.listdir(folder_path):
        if file.endswith(".ckpt"):
            ckpts.append(file)
    if len(ckpts) == 0:
        return None

    model_hparams = {"t": 4}
    optimizer_hparams={"lr": 0.1}
    model = Module(model_hparams, optimizer_hparams)
    state_dict = torch.load(os.path.join(folder_path, ckpts[-1]))["state_dict"]
    model.load_state_dict(state_dict)
    return ckpts[-1], model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    path = 'data/'

    model_checkpoint_folder = "models\\lightning_logs\\version_0\\checkpoints"

    event_handler = FileChangeHandler(model_checkpoint_folder)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()