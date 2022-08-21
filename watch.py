import sys
import time
import sched
import random
import re
import os
import csv
import random

import argparse
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
    def __init__(self, data_path, model_checkpoint_folder, args):
        self.data_path = data_path
        self.model_checkpoint_folder = model_checkpoint_folder
        self.latest_model_file = None
        self.iter = 0
        self.dir_name = None
        self.prev_id = None
        self.model = None

        #Other args
        self.reload_model_every_n_iterations = args.reload

    def load_model(self):
        model_file = train.get_latest_model(self.model_checkpoint_folder)

        if not model_file == self.latest_model_file and not model_file == None:
            print('Loading new model!')
            model_hparams = {"t": 4}
            optimizer_hparams={"lr": 0.1}
            self.model = Module(data_path, model_hparams, optimizer_hparams)
            state_dict = torch.load(model_file)["state_dict"]
            self.model.load_state_dict(state_dict)
            self.latest_model_file = model_file

        print(f'Iteration {self.iter}\tLoaded model {self.latest_model_file}\tDeterministic: {self.iter % 4 >= 2}')

    def on_created(self, event):
        if event.is_directory:
            output_file = os.path.join(event.src_path, 'outputFile.csv')
            if not os.path.exists(output_file):
                file = open(output_file, 'w')
                file.write("id,A,B,X,Y,Up,Down,Left,Right\n")
                file.close()
            if self.iter % self.reload_model_every_n_iterations == 0:
                self.load_model()
            self.iter += 1
        else:
            event_dir_name = os.path.dirname(event.src_path)
            event_base_name = os.path.basename(event.src_path)

            if not event.src_path.split('.')[1] == "csv":
                id = int(re.sub("[^0-9]", "", event_base_name))
                output_file = event_dir_name + "/outputFile.csv"

                #Sometimes FIleSystemWatcher triggers twice, so check if new id is higher than previous id,
                #or if it is a new directory (in that case, reset the previous id)
                new_dir = False
                if not event_dir_name == self.dir_name:
                    new_dir = True
                    self.prev_id = 0
                self.dir_name = event_dir_name

                if id <= self.prev_id:
                    return #ignore false screenshot
                else:
                    self.prev_id = id
                if self.model:
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
                        input["screenshots"] = [f"{event_dir_name}/screenshot_{point['id']}.png" for point in input["previous_points"]]

                    #Add current frame
                    input["screenshots"].append(f"{event_dir_name}/screenshot_{id}.png")

                    next_inputs = train.get_next_input(self.model, input, deterministic=self.iter % 4 >= 2)

                    add_output_to_file(output_file, id, next_inputs)
                else:
                    #If no model, return random input
                    i = random.randint(0, 63)
                    next_inputs = [int(i/32)%2==1, int(i/16)%2==1, int(i/8)%2==1, int(i/4)%2==1, i%4==0, i%4==1, i%4==2, i%4==3]
                    next_inputs = [int(x) for x in next_inputs]

                    add_output_to_file(output_file, id, next_inputs)

    def on_modified(self, event):
        pass

if __name__ == "__main__":
    print('Running file watch.py')
    parser = argparse.ArgumentParser()

    # Experiment number, required!
    parser.add_argument('-e', '--experiment', type=int, required=True)
    parser.add_argument('-r', '--reload', type=int, default=2)

    args = parser.parse_args()

    data_path = f'experiments\\experiment_{args.experiment}\\data'
    model_checkpoint_folder = f"experiments\\experiment_{args.experiment}\\models\\lightning_logs"

    event_handler = FileChangeHandler(data_path, model_checkpoint_folder, args)
    observer = Observer()
    observer.schedule(event_handler, data_path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()