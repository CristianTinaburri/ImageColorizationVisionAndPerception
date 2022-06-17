import torch
import torch.nn as nn

class autoencoder(nn.Module):
    def __init__(self, dimension, features_maps, percentage_dropout):
        super(autoencoder, self).__init__()
        
        self.initial_block = nn.Sequential(
          nn.Conv2d(1, features_maps, kernel_size=4, stride=2, padding=1),
          nn.BatchNorm2d(features_maps),
          nn.ReLU(),
          nn.Dropout(percentage_dropout)
        )

        self.down_block_1 = nn.Sequential(
          nn.Conv2d(features_maps, int(features_maps*2), kernel_size=4, stride=2, padding=1),
          nn.BatchNorm2d(int(features_maps*2)),
          nn.ReLU(),
          nn.Dropout(percentage_dropout)
        )

        self.down_block_2 = nn.Sequential(
          nn.Conv2d(int(features_maps*2), int(features_maps*4), kernel_size=4, stride=2, padding=1),
          nn.BatchNorm2d(int(features_maps*4)),
          nn.ReLU(),
          nn.Dropout(percentage_dropout)
        )

        self.down_block_3 = nn.Sequential(
          nn.Conv2d(int(features_maps*4), int(features_maps*8), kernel_size=4, stride=2, padding=1),
          nn.BatchNorm2d(int(features_maps*8)),
          nn.ReLU(),
          nn.Dropout(percentage_dropout)
        )

        self.up_block_1 = nn.Sequential(
          nn.ConvTranspose2d(int(features_maps*8), int(features_maps*4), kernel_size=4, stride=2, padding=1),
          nn.BatchNorm2d(int(features_maps*4)),
          nn.ReLU(),
          nn.Dropout(percentage_dropout)
        )

        self.up_block_2 = nn.Sequential(
          nn.ConvTranspose2d(int(features_maps*4), int(features_maps*2), kernel_size=4, stride=2, padding=1),
          nn.BatchNorm2d(int(features_maps*2)),
          nn.ReLU(),
          nn.Dropout(percentage_dropout)
        )

        self.up_block_3 = nn.Sequential(
          nn.ConvTranspose2d(int(features_maps*2), int(features_maps), kernel_size=4, stride=2, padding=1),
          nn.BatchNorm2d(int(features_maps)),
          nn.ReLU(),
          nn.Dropout(percentage_dropout)
        )

        self.output_block = nn.Sequential(
          nn.ConvTranspose2d(features_maps, 2, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, input):
        # down
        x = self.initial_block(input)
        x = self.down_block_1(x)
        x = self.down_block_2(x)
        x = self.down_block_3(x)

        # down
        x = self.up_block_1(x)
        x = self.up_block_2(x)
        x = self.up_block_3(x)

        x = self.output_block(x)

        return x