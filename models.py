from itertools import count
from turtle import speed
from typing import Concatenate
import torch
import torch.nn as nn
import torch.nn.functional as F

all_objectives = {
    'speed': 5,
    'death': 2
}

class BaseMarioModel(nn.Module):
    def __init__(self, t=4, objectives=['speed', 'death']):
        super().__init__()
        act_fn = nn.ReLU
        self.t = t

        self.objectives = objectives
        self.outputs = 0
        for objective in self.objectives:
            if objective in all_objectives.keys():
                self.outputs += all_objectives[objective]

        self.input_size_per_image = 64*64*3
        self.input_size_per_action = 8
        self.convnet = nn.Sequential(
            nn.Conv2d(t*3, 8, 3, stride=2, padding=1), #64x64
            act_fn(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),  #32x32
            act_fn(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  #16x16
            act_fn(),
            nn.Flatten()
        )

        self.linear = nn.Sequential(
            nn.Linear(8*8*32 + t*8, 128),
            act_fn(),
            nn.Linear(128, self.outputs)
        )
    
    def seperate_output(self, output):
        output_tensors = {}

        count_output = 0
        for objective in self.objectives:
            objective_size = all_objectives[objective]
            output_tensor = output[:, count_output:count_output + objective_size]
            output_tensors[objective] = output_tensor
        return output_tensors

    def forward(self, screenshot_tensor, previous_actions_tensor):
        b, t, h, w, c = screenshot_tensor.size()
        screenshot_tensor = screenshot_tensor.reshape(b, t*c, h, w)
        screenshot_tensor = self.convnet(screenshot_tensor)

        previous_actions_tensor = torch.flatten(previous_actions_tensor, start_dim=1)
        concatenated = torch.cat((screenshot_tensor, previous_actions_tensor), dim=1)

        output = self.linear(concatenated)

        return output

    def is_correct_size(self, screenshot_tensor, previous_actions_tensor):
        _, t, h, w, c = screenshot_tensor.size()
        screenshot_correct = t == self.t and h * w * c == self.input_size_per_image

        _, t, c = previous_actions_tensor.size()
        previous_action_correct = t == self.t and c == self.input_size_per_action

        return screenshot_correct and previous_action_correct