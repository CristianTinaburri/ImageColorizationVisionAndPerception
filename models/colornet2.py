import torch
import torch.nn as nn
import torch.nn.functional as F

class colornet2(nn.Module):
    def __init__(self, d=128):
        super(colornet2, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1) # out: 32 x 16 x 16
        self.conv1_bn = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(p=0.5)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1) # out: 64 x 8 x 8
        self.conv2_bn = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # out: 128 x 4 x 4
        self.conv3_bn = nn.BatchNorm2d(128)

        '''self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) # out: 128 x 4 x 4
        self.conv4_bn = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) # out: 128 x 4 x 4
        self.conv5_bn = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) # out: 128 x 4 x 4
        self.conv6_bn = nn.BatchNorm2d(128)'''

        self.tconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # out: 64 x 8 x 8
        self.tconv1_bn = nn.BatchNorm2d(64)

        self.tconv2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1) # out: 32 x 16 x 16
        self.tconv2_bn = nn.BatchNorm2d(64)

        self.tconv3 = nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1) # out: 2 x 32 x 32

    def forward(self, input):
        x = self.dropout(F.relu(self.conv1_bn(self.conv1(input))))
        x = self.dropout(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.dropout(F.relu(self.conv3_bn(self.conv3(x))))
        '''x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6_bn(self.conv6(x)))'''
        x = self.dropout(F.relu(self.tconv1_bn(self.tconv1(x))))
        x = self.dropout(F.relu(self.tconv2_bn(self.tconv2(x))))
        x = self.tconv3(x)

        return x