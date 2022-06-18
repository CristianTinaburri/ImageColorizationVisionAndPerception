import torch
import torch.nn as nn
import torchvision.models as models

class ResNetBackbone18(nn.Module):
  def __init__(self, p=0.5):
    super(ResNetBackbone18, self).__init__()

    self.percentage_dropout = p

    resnet = models.resnet18(pretrained=True, num_classes=1000) 

    resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 

    self.resnet_backbone = nn.Sequential(*list(resnet.children())[0:6])

    for param in self.midlevel_resnet.parameters():
      param.requires_grad = False

    self.custom_layer = nn.Sequential(     
      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),

      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Dropout(0.5),

      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2),

      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Dropout(0.5),
      
      nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=0),
      nn.ConvTranspose2d(2, 2, kernel_size=2, stride=2),
    )

  def forward(self, input):

    resnet_backbone = self.resnet_backbone(input)

    output = self.custom_layer(midlevel_features)
    
    return output