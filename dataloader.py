import datetime
from doctest import UnexpectedException
from genericpath import isfile
from operator import contains
import time
import os
import sys

import numpy as np
import torch
import csv
import cv2

from torch.utils.data import Dataset, DataLoader, random_split
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
    def __init__(self, root='data/', t=4, verbose=False):
        self.data = []
        for data in os.listdir(root):
            data_points = []
            state_file = os.path.join(root, data, 'stateFile.csv')
            output_file = os.path.join(root, data, 'outputFile.csv')

            with open(output_file, newline='') as state_file_data:
                reader = csv.DictReader(state_file_data, delimiter=',')

                previous_points = []

                for row in reader:
                    previous_points.append(row)
                    if len(previous_points) > t:
                        previous_points = previous_points[1:]
                    if len(previous_points) == t:
                        data_point = {}
                        data_point["previous_points"] = previous_points.copy()
                        data_point["screenshots"] = [os.path.join(root, data, f'screenshot_{point["id"]}.png') for point in data_point["previous_points"]]
                        data_points.append(data_point)

            with open(state_file, newline='') as state_file_data:
                reader = csv.DictReader(state_file_data, delimiter=',')

                i=0
                for row in reader:
                    try:
                        last_id = int(data_points[i]["previous_points"][t-1]['id'])
                    except IndexError:
                        break
                    
                    if int(row['id']) == last_id:
                        if i > 0:
                            data_points[i-1]['future_state'] = row.copy()
                        data_points[i]['current_state'] = row.copy()
                        i += 1

            self.data.extend(data_points[:-1])
        if verbose:
            print(f'Created {len(self.data)} items in dataset')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        screenshot_tensor, previous_actions_tensor = input_to_tensors(item['screenshots'], item['previous_points'])

        value = 0
        value = int(item['future_state']['MarioX']) - int(item['current_state']['MarioX'])
        return screenshot_tensor, previous_actions_tensor, value

def input_to_tensors(screenshots, previous_points, append_screenshots_to=4, append_previous_actions_to=3):
    screenshot_tensors = []
    for screenshot_path in screenshots:
        if os.path.isfile(screenshot_path):
            pic = plt.imread(screenshot_path)
            pic = cv2.resize(pic, (64, 64))
            pic_tensor = torch.from_numpy(pic)[:,:,:3]
            screenshot_tensors.append(pic_tensor)
    if len(screenshot_tensors) == 0:
        raise NotImplementedError()
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

def create_dataloader(batch_size = 32, num_workers=1):
    dataset = OfflineMarioDataset(verbose=False)
    train_size = test_size = int(len(dataset) / 3)
    val_size = len(dataset) - train_size - test_size
    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    return train_loader, test_loader, val_loader