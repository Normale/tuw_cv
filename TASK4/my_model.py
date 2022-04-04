# TUWIEN - WS2021 CV: Task4 - Mask Classification using CNN
# *********+++++++++*******++++Group_9-11_11

from typing import List
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F

# A simple CNN for mask classification


class MaskClassifier(nn.Module):

    def __init__(self, name, img_size=64, dropout: float = 0, batch_norm: bool = False):
        # Initializes the network architecture, creates a simple cnn of convolutional and max pooling
        # layers. If batch_norm is set to true, batchnorm is applied to the convolutional layers.
        # If dropout>0, dropout is applied to the linear layers.
        # HINT: nn.Conv2d(...), nn.MaxPool2d(...), nn.BatchNorm2d(...), nn.Linear(...), nn.Dropout(...)
        # dropout: dropout rate between 0 and 1
        # batch_norm: if batch normalization should be applied
        # student code start
        super(MaskClassifier, self).__init__()
        self.dropout = dropout
        self.name = name
        
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32) if batch_norm else nn.Identity(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=self.dropout),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32) if batch_norm else nn.Identity(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=self.dropout),

            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(128) if batch_norm else nn.Identity(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(p=self.dropout),

            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(128) if batch_norm else nn.Identity(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(p=self.dropout),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(32*14*14,1),
            # nn.Linear(4*4*128, 512), 
            # in case you wonder wtf is (4*4*128) :
            # 4 * 4 is image after 4 MaxPool2d with kernel_size_2
            # it was initially 64, after first its 32x32
            # second: 16x16, third 8x8, fourth 4x4
            # 128 is output of convolution (we are convolving from 3 -> 32 -> 64 -> 128)
            # my understanding might be wrong, but it works ¯\_(ツ)_/¯
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=self.dropout),
            # nn.Linear(512, 256),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=self.dropout),
            # nn.Linear(256,10),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=self.dropout),
            # nn.Linear(10,1),
            nn.Sigmoid()
            # Going further, output sizes dont really matter,
            # but the end should be positive between 0-1,
            # because BCELoss is used to test 
            # Sigmoid activation seems to work fine
            # https://pytorch.org/docs/1.9.1/generated/torch.nn.BCELoss.html 
        )
        # student code end

    def forward(self, x: Tensor):
        # Applies the predefined layers of the network to x
        # x: input tensor to be classified [batch_size x channels x height x width] - Tensor
        # student code start
        # Convolution & Pool Layers
        x = self.layers(x)
        x = x.view(-1, 32*14*14)
        x = self.linear_layers(x)
        # print(x)
        # student code end
        return x
