import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNet(nn.Module):
  def __init__(self):
    super(EfficientNet, self).__init__()

    efficientnet = models.efficientnet_b6(pretrained=True) 

    efficientnet.conv1.weight = nn.Parameter(efficientnet.conv1.weight.sum(dim=1).unsqueeze(1)) 

    self.midlevel_resnet = nn.Sequential(*list(efficientnet.children())[:-1])

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
      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
      nn.Upsample(scale_factor=2)
    )

  def forward(self, input):

    midlevel_features = self.midlevel_resnet(input)

    output = self.upsample(midlevel_features)
    return midlevel_features