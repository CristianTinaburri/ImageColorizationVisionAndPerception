import torch
import torch.nn as nn
import torch.nn.functional as F

class colorfulcolorization(nn.Module):
    def __init__(self):
        super(colorfulcolorization, self).__init__()

        self.colorful = nn.Sequential(     
          nn.Conv2d(1, 64, stride=1, kernel_size=1, dilation=1),
          nn.ReLU(),
          nn.Conv2d(64, 64, stride=2, kernel_size=1, dilation=1),
          nn.ReLU(),
          nn.BatchNorm2d(64),

          nn.Conv2d(64, 128, stride=1, kernel_size=1, dilation=2),
          nn.ReLU(),
          nn.Conv2d(128, 128, stride=2, kernel_size=1, dilation=2),
          nn.ReLU(),
          nn.BatchNorm2d(128),

          nn.Conv2d(128, 256, stride=1, kernel_size=1, dilation=4),
          nn.ReLU(),
          nn.Conv2d(256, 256, stride=1, kernel_size=1, dilation=4),
          nn.ReLU(),
          nn.Conv2d(256, 256, stride=2, kernel_size=1, dilation=4),
          nn.ReLU(),
          nn.BatchNorm2d(256),

          nn.Conv2d(256, 512, stride=1, kernel_size=1, dilation=8),
          nn.ReLU(),
          nn.Conv2d(512, 512, stride=1, kernel_size=1, dilation=8),
          nn.ReLU(),
          nn.Conv2d(512, 512, stride=1, kernel_size=1, dilation=8),
          nn.ReLU(),
          nn.BatchNorm2d(512),

          nn.Conv2d(512, 512, stride=1, kernel_size=2, dilation=16),
          nn.ReLU(),
          nn.Conv2d(512, 512, stride=1, kernel_size=2, dilation=16),
          nn.ReLU(),
          nn.Conv2d(512, 512, stride=1, kernel_size=2, dilation=16),
          nn.ReLU(),
          nn.BatchNorm2d(512),

          nn.Conv2d(512, 512, stride=1, kernel_size=2, dilation=16),
          nn.ReLU(),
          nn.Conv2d(512, 512, stride=1, kernel_size=2, dilation=16),
          nn.ReLU(),
          nn.Conv2d(512, 512, stride=1, kernel_size=2, dilation=16),
          nn.ReLU(),
          nn.BatchNorm2d(512),

          nn.Conv2d(512, 256, stride=1, kernel_size=1, dilation=8),
          nn.ReLU(),
          nn.Conv2d(256, 256, stride=1, kernel_size=1, dilation=8),
          nn.ReLU(),
          nn.Conv2d(256, 256, stride=1, kernel_size=1, dilation=8),
          nn.ReLU(),
          nn.BatchNorm2d(256),

          nn.Conv2d(256, 128, stride=0.5, kernel_size=1, dilation=4),
          nn.ReLU(),
          nn.Conv2d(128, 128, stride=1, kernel_size=1, dilation=4),
          nn.ReLU(),
          nn.Conv2d(128, 128, stride=1, kernel_size=1, dilation=4),
          nn.ReLU(),
          nn.BatchNorm2d(128),
        )


    def forward(self, x):
        output = self.colorful(x)

        return output