import torch
import torch.nn as nn
import torchvision.models as models

class ResNetBackbone152(nn.Module):
  def __init__(self):
    super(ResNetBackbone152, self).__init__()

    resnet = models.resnet152(pretrained=True, num_classes=1000) 

    resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 

    self.midlevel_resnet = nn.Sequential(*list(resnet.children())[:-2])

    for param in self.midlevel_resnet.parameters():
      param.requires_grad = False

    self.upsample = nn.Sequential(     
      nn.Conv2d(2048, 8, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(8),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1),
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
      nn.Upsample(scale_factor=8)
    )

  def forward(self, input):

    midlevel_features = self.midlevel_resnet(input)

    output = self.upsample(midlevel_features)
    
    return output