from inspect import stack
from itertools import count
from turtle import forward, speed
from types import SimpleNamespace
from typing import Concatenate
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, c_in, act_fn, subsample=False, c_out=-1):
        """
        Inputs:
            c_in - Number of input features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        if not subsample:
            c_out = c_in

        # Network representing F
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False),  # No bias needed as the Batch Norm handles it
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out)
        )

        # 1x1 convolution with stride 2 means we take the upper left value, and transform it to new output size
        self.downsample = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2) if subsample else None
        self.act_fn = act_fn()

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        out = self.act_fn(out)
        return out


class StupidModel(nn.Module):
    def __init__(self, t=4, num_outputs=7):
        super().__init__()

        act_fn = nn.ReLU
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
            nn.Linear(128, num_outputs)
        )
    
    def forward(self, screenshot_tensor, previous_actions_tensor):
        b, t, h, w, c = screenshot_tensor.size()
        screenshot_tensor = screenshot_tensor.reshape(b, t*c, h, w)
        screenshot_tensor = self.convnet(screenshot_tensor)

        previous_actions_tensor = torch.flatten(previous_actions_tensor, start_dim=1)
        concatenated = torch.cat((screenshot_tensor, previous_actions_tensor), dim=1)

        output = self.linear(concatenated)

        return output

class ResNetModel(nn.Module):
    def __init__(self, t=4, num_outputs=7, num_blocks=[3,3,3,3], c_hidden=[16,32,64,128], **kwargs):
        """
        Inputs:
            t - number of previous frames (first layer has (3+8) * t channels)
            num_outputs - Number of classification outputs (10 for CIFAR10)
            num_blocks - List with the number of ResNet blocks to use. The first block of each group uses downsampling, except the first.
            c_hidden - List with the hidden dimensionalities in the different blocks. Usually multiplied by 2 the deeper we go.
        """
        super().__init__()
        self.hparams = SimpleNamespace(input_channels=(3+8)*t,
                                       num_outputs=num_outputs,
                                       c_hidden=c_hidden,
                                       num_blocks=num_blocks,
                                       act_fn_name='relu',
                                       act_fn=nn.ReLU,
                                       block_class=ResNetBlock)
        self._create_network()
        self._init_params()

    def _create_network(self):
        c_hidden = self.hparams.c_hidden

        # A first convolution on the original image to scale up the channel size
        self.input_net = nn.Sequential(
            nn.Conv2d(self.hparams.input_channels, c_hidden[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_hidden[0]),
            self.hparams.act_fn()
        )

        # Creating the ResNet blocks
        blocks = []
        for block_idx, block_count in enumerate(self.hparams.num_blocks):
            for bc in range(block_count):
                subsample = (bc == 0 and block_idx > 0) # Subsample the first block of each group, except the very first one.
                blocks.append(
                    self.hparams.block_class(c_in=c_hidden[block_idx if not subsample else (block_idx-1)],
                                             act_fn=self.hparams.act_fn,
                                             subsample=subsample,
                                             c_out=c_hidden[block_idx])
                )
        self.blocks = nn.Sequential(*blocks)

        # Mapping to classification output
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(c_hidden[-1], self.hparams.num_outputs)
        )

    def _init_params(self):
        # Based on our discussion in Tutorial 4, we should initialize the convolutions according to the activation function
        # Fan-out focuses on the gradient distribution, and is commonly used in ResNets
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, screenshot_tensor, previous_actions_tensor):
        b, t, h, w, c = screenshot_tensor.size()
        screenshot_tensor = screenshot_tensor.reshape(b, t*c, h, w)
        previous_actions_tensor = previous_actions_tensor.reshape(b, t*8, 1, 1)
        previous_actions_tensor = previous_actions_tensor.repeat(1, 1, h, w)
        stacked_input_tensor = torch.cat((screenshot_tensor, previous_actions_tensor), 1)

        x = self.input_net(stacked_input_tensor)
        x = self.blocks(x)
        output = self.output_net(x)
        return output

class BaseMarioModel(nn.Module):
    def __init__(self, t=4, objectives=['speed', 'death']):
        super().__init__()
        self.t = t
        self.input_size_per_image = 64*64*3
        self.input_size_per_action = 8

        self.objectives = objectives
        self.outputs = 0
        for objective in self.objectives:
            if objective in all_objectives.keys():
                self.outputs += all_objectives[objective]

        self.forwardModel = ResNetModel(t=t, num_outputs=self.outputs)
    
    def seperate_output(self, output):
        output_tensors = {}

        count_output = 0
        for objective in self.objectives:
            objective_size = all_objectives[objective]
            output_tensor = output[:, count_output:count_output + objective_size]
            output_tensors[objective] = output_tensor
        return output_tensors

    def forward(self, screenshot_tensor, previous_actions_tensor):
        return self.forwardModel(screenshot_tensor, previous_actions_tensor)

    def is_correct_size(self, screenshot_tensor, previous_actions_tensor):
        _, t, h, w, c = screenshot_tensor.size()
        screenshot_correct = t == self.t and h * w * c == self.input_size_per_image

        _, t, c = previous_actions_tensor.size()
        previous_action_correct = t == self.t and c == self.input_size_per_action

        return screenshot_correct and previous_action_correct

      
#Some globals to be used in the entire project
all_objectives = {
    'speed': 5,
    'death': 2
}  

all_models = {
    'stupid_model': StupidModel,
    'res_net_model': ResNetModel
}