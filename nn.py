import datetime
import time
import os
import sys

import numpy as np
import torch
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image

import matplotlib.pyplot as plt

"""Each datapoint consist of the following:

Data:
The previous t screenshots in 64x64 grayscale (t, 64, 64)
The generated button presses on those screenshot;
boolean for A,B,X,Y + one hot encoding for up/down/left/right (t, 4) + (t, 4)

Label:
Value score generated using the StateFile, currently set to delta(marioX_curr, marioX_start),
where marioX_end is 10 frames after the last screenshot to determine the effect of the last button presses
"""
class OfflineMarioDataset(Dataset):
    def __init__(self, root='data/', t=4, verbose=False):
        self.data = []
        for data in os.listdir(root):
            dataPoints = []
            stateFile = os.path.join(root, data, 'stateFile.csv')
            outputFile = os.path.join(root, data, 'outputFile.csv')

            with open(outputFile, newline='') as stateFileData:
                reader = csv.DictReader(stateFileData, delimiter=',')

                previousPoints = []

                for row in reader:
                    previousPoints.append(row)
                    if len(previousPoints) > t:
                        previousPoints = previousPoints[1:]
                    if len(previousPoints) == t:
                        dataPoint = {}
                        dataPoint["previousPoints"] = previousPoints.copy()
                        dataPoint["screenshots"] = [os.path.join(root, data, f'screenshot_{point["id"]}.png') for point in dataPoint["previousPoints"]]
                        dataPoints.append(dataPoint)

            with open(stateFile, newline='') as stateFileData:
                reader = csv.DictReader(stateFileData, delimiter=',')

                i=0
                for row in reader:
                    try:
                        lastID = int(dataPoints[i]["previousPoints"][t-1]['id'])
                    except IndexError:
                        break
                    
                    if int(row['id']) == lastID:
                        if i > 0:
                            dataPoints[i-1]['futureState'] = row.copy()
                        dataPoints[i]['currentState'] = row.copy()
                        i += 1

            self.data.extend(dataPoints)
        if verbose:
            print(f'Created {len(self.data)} items in dataset')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """TODO: return processed item (meaning, just return tensors)
        img_path, class_name = self.data[idx]
        pic = plt.imread(img_path)
        pic = cv2.resize(pic, (256, 256))

        pic_tensor = torch.from_numpy(pic)
        return pic_tensor"""

if __name__ == '__main__':
    dataset = OfflineMarioDataset(verbose=True)