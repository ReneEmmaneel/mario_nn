from typing import Concatenate
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseMarioModel(nn.Module):
    def __init__(self, t=4):
        super().__init__()
        act_fn = nn.ReLU

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
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, screenshot_tensor, previous_actions_tensor):
        b, t, h, w, c = screenshot_tensor.size()
        screenshot_tensor = screenshot_tensor.reshape(b, t*c, h, w)
        screenshot_tensor = self.convnet(screenshot_tensor)

        previous_actions_tensor = torch.flatten(previous_actions_tensor, start_dim=1)
        concatenated = torch.cat((screenshot_tensor, previous_actions_tensor), dim=1)

        output = self.linear(concatenated)

        return output