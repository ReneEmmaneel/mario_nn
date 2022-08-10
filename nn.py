import datetime
from operator import contains
import time
import os
import sys

import numpy as np
import torch
import csv
import cv2

from torch.utils.data import Dataset, DataLoader
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

            self.data.extend(data_points)
        if verbose:
            print(f'Created {len(self.data)} items in dataset')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        valid = 'future_state' in item.keys()

        screenshot_tensors = []
        for screenshot_path in item['screenshots']:
            pic = plt.imread(screenshot_path)
            pic = cv2.resize(pic, (64, 64))
            pic_tensor = torch.from_numpy(pic)[:,:,:3]
            screenshot_tensors.append(pic_tensor)
        screenshot_tensor = torch.stack(screenshot_tensors, dim=0)

        previous_actions = []
        for previous_action in item["previous_points"]:
            previous_actions.append(torch.tensor([int(v) for v in previous_action.values()][1:]))
        previous_actions_tensor = torch.stack(previous_actions, dim=0)

        value = 0
        if 'future_state' in item.keys():
            value = int(item['future_state']['MarioX']) - int(item['current_state']['MarioX'])
        return screenshot_tensor, previous_actions_tensor, value, valid

if __name__ == '__main__':
    dataset = OfflineMarioDataset(verbose=True)
    loader = DataLoader(dataset, batch_size=2)

    for i, sample in enumerate(loader):
        print(i)
        print(sample)