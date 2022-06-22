import torch
import torch.nn as nn
import torchvision.models as models

class basemodel(nn.Module):
  def __init__(self, p=0.5):
    super(basemodel, self).__init__()

    self.percentage_dropout = p

    self.base_model = nn.Sequential(     
      nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.ConvTranspose2d(64, 64, kernel_size=1, stride=1),

      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Dropout(0.5),

      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.ConvTranspose2d(32, 64, kernel_size=1, stride=1),

      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Dropout(0.5),
      
      nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
      nn.ConvTranspose2d(2, 2, kernel_size=1, stride=1),
    )

  def forward(self, input):

    output = self.base_model(input)
    
    return output