import datetime
from doctest import UnexpectedException
from genericpath import isfile
from math import floor
from operator import contains
import time
import os
import sys
from turtle import speed

import numpy as np
import torch
import csv
import cv2

from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision.utils import make_grid, save_image

import matplotlib.pyplot as plt

"""Each data_point consist of the following:

Data:
The previous t screenshots in 64x64 grayscale (t, 64, 64)
The generated button presses on those screenshot;
boolean for A,B,X,Y,up,down,left,right (t, 8)

Label:
Value score generated using the State_file, currently set to delta(marioX_curr, marioX_start),
where marioX_end is 10 frames after the last screenshot to determine the effect of the last button presses
"""
class OfflineMarioDataset(Dataset):
    def __init__(self, root='data/', t=4, objectives=['speed', 'death']):
        self.data = []
        self.dirs = []
        self.root = root
        self.t = t
        self.objectives = objectives
        self.remake = False

        self.update_data()
    
    def update_data(self):
        if self.remake: #If set to true, remake the entire dataset by deleting all existing data
            self.data = []
            self.dirs = []

        for data in os.listdir(self.root)[:-1]:
            if not data in self.dirs: #Skip the last one, as the data for that run is still coming in
                self.add_dir_to_data_list(data)
            self.remake = False

        if len(self.dirs) == 0:
            #Load the not finished first run
            for data in os.listdir(self.root):
                self.add_dir_to_data_list(data)
                print(f'Added {data} to dataset')
            self.remake = True

    def add_dir_to_data_list(self, dir_name):
        data_points = []
        state_file = os.path.join(self.root, dir_name, 'stateFile.csv')
        output_file = os.path.join(self.root, dir_name, 'outputFile.csv')

        if not os.path.isfile(state_file) or not os.path.isfile(output_file):
            return

        with open(output_file, newline='') as state_file_data:
            reader = csv.DictReader(state_file_data, delimiter=',')

            previous_points = []

            for row in reader:
                previous_points.append(row)
                if len(previous_points) > self.t:
                    previous_points = previous_points[1:]
                if len(previous_points) == self.t:
                    data_point = {}
                    data_point["previous_points"] = previous_points.copy()
                    data_point["screenshots"] = [os.path.join(self.root, dir_name, f'screenshot_{point["id"]}.png') for point in data_point["previous_points"]]
                    data_points.append(data_point)

        with open(state_file, newline='') as state_file_data:
            reader = csv.DictReader(state_file_data, delimiter=',')

            i=0
            for row in reader:
                try:
                    last_id = int(data_points[i]["previous_points"][self.t-1]['id'])
                except IndexError:
                    break
                
                if int(row['id']) == last_id:
                    if i > 0:
                        data_points[i-1]['future_state'] = row.copy()
                    data_points[i]['current_state'] = row.copy()
                    i += 1

        for data_point in data_points:
            if not "future_state" in data_point:
                data_point["future_state"] = {
                    "MarioX": data_point["current_state"]["MarioX"],
                    "MarioState": '9'
                }

        def get_values(item):
            values_list = []
            if 'speed' in self.objectives:
                values_list.append(self.speed_to_encoding(int(item['future_state']['MarioX']) - int(item['current_state']['MarioX'])))
            if 'death' in self.objectives:
                values_list.append(int(item['future_state']['MarioState'] == '9'))
            return values_list

        data_points = [{'input': input_to_tensors(x['screenshots'], x['previous_points']), 'values': get_values(x)} for x in data_points]
        self.data.extend(data_points)
        self.dirs.append(dir_name)

    def speed_to_encoding(self, value):
        #range -inf to -16: 0
        #       -15 to  -6: 1
        #        -5 to   4: 2
        #         5 to   4: 3
        #        15 to inf: 4
        speed_rounded = floor((value + 5) / 10) + 2
        speed_rounded = max(0, min(speed_rounded, 4))
        return speed_rounded

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        screenshot_tensor, previous_actions_tensor = item['input']
        values = item['values']
        
        return screenshot_tensor, previous_actions_tensor, values

def input_to_tensors(screenshots, previous_points, append_screenshots_to=4, append_previous_actions_to=3):
    screenshot_tensors = []
    for screenshot_path in screenshots:
        if os.path.isfile(screenshot_path):
            try:
                pic = plt.imread(screenshot_path)
                pic = cv2.resize(pic, (64, 64))
                pic_tensor = torch.from_numpy(pic)[:,:,:3]
                screenshot_tensors.append(pic_tensor)
            except SyntaxError:
                print(f'Error loading screenshot with path {screenshot_path}')
    if len(screenshot_tensors) == 0:
        screenshot_tensors.append(torch.zeros(64, 64, 3))
    while len(screenshot_tensors) < append_screenshots_to:
        screenshot_tensors.insert(0, screenshot_tensors[0])
    screenshot_tensor = torch.stack(screenshot_tensors, dim=0)

    previous_actions = []
    for previous_action in previous_points:
        previous_actions.append(torch.tensor([int(v) for v in previous_action.values()][1:]))
    while len(previous_actions) < append_previous_actions_to:
        previous_actions.insert(0, torch.zeros(8))
    previous_actions_tensor = torch.stack(previous_actions, dim=0)
    return screenshot_tensor, previous_actions_tensor

def loaders_from_dataset(dataset, batch_size = 32, num_workers=1, weighted=False):
    if not weighted:
        train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        return train_loader
    else:
        labels = torch.tensor([t[2][0] for t in dataset])
        class_sample_count = np.array([len(np.where(labels == t)[0]) for t in range(5)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t[2][0]] for t in dataset])

        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
        return train_loader

def create_dataloader(batch_size = 32, num_workers=1, weighted=False, root='data/', t=4, objectives=['speed', 'death']):
    dataset = OfflineMarioDataset(root=root, t=t, objectives=objectives)

    if len(dataset) == 0:
        return None

    return dataset, loaders_from_dataset(dataset, batch_size=batch_size, num_workers=num_workers, weighted=weighted)

def update_dataloader(dataset, batch_size = 32, num_workers=1, weighted=False):
    dataset.update_data()

    if len(dataset) == 0:
        return None

    return dataset, loaders_from_dataset(dataset, batch_size=batch_size, num_workers=num_workers, weighted=weighted)

if __name__ == '__main__':
    dataset, dataloader = create_dataloader(root='experiments\experiment_1\data')