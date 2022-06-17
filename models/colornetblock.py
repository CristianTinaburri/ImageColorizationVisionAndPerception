import torch
import torch.nn as nn

class colornetblock(nn.Module):
    def __init__(self, dim, percentage_dropout):
        super(colornetblock, self).__init__()
        
        self.initial_block = nn.Sequential(
          nn.Conv2d(1, dim, kernel_size=4, stride=2, padding=1),
          nn.BatchNorm2d(dim),
          nn.ReLU(),
          nn.Dropout(percentage_dropout)
        )

        self.pre_block = nn.Sequential(
          nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1),
          nn.BatchNorm2d(dim),
          nn.ReLU(),
          nn.Dropout(percentage_dropout)
        )

        self.up_block = nn.Sequential(
          nn.Conv2d(dim, dim*2, kernel_size=4, stride=2, padding=1),
          nn.BatchNorm2d(dim*2),
          nn.ReLU(),
          nn.Dropout(percentage_dropout)
        )

        self.same_block = nn.Sequential(
          nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(dim*2),
          nn.ReLU(),
          nn.Dropout(percentage_dropout)
        )

        self.down_block = nn.Sequential(
          nn.ConvTranspose2d(dim*2, dim, kernel_size=4, stride=2, padding=1),
          nn.BatchNorm2d(dim),
          nn.ReLU(),
          nn.Dropout(percentage_dropout)
        )

        self.end_block = nn.Sequential(
          nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1),
          nn.BatchNorm2d(dim),
          nn.ReLU(),
          nn.Dropout(percentage_dropout)
        )

        self.output_block = nn.Sequential(
          nn.ConvTranspose2d(dim, 2, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, input):
        # up
        x = self.initial_block(input)
        x = self.pre_block(x)
        x = self.up_block(x)

        # deep
        x = self.same_block(x)
        x = self.same_block(x)
        x = self.same_block(x)

        # down
        x = self.down_block(x)
        x = self.end_block(x)
        x = self.output_block(x)

        return x