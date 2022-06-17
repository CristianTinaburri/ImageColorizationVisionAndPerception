import torch
import torch.nn as nn
import torchvision.models as models

class ResNetBackbone18(nn.Module):
  def __init__(self, p=0.5):
    super(ResNetBackbone18, self).__init__()

    self.percentage_dropout = p

    resnet = models.resnet18(pretrained=True, num_classes=1000) 

    resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 

    self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])

    for param in self.midlevel_resnet.parameters():
      param.requires_grad = False

    self.upsample = nn.Sequential(     
      nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Dropout(self.percentage_dropout),
      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Dropout(self.percentage_dropout),
      nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
      nn.Upsample(scale_factor=2)
    )

  def forward(self, input):

    midlevel_features = self.midlevel_resnet(input)

    output = self.upsample(midlevel_features)
    
    return output