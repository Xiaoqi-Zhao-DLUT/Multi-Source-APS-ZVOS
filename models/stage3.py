import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import resnet34

class stage3(nn.Module):

    def __init__(self):
        super(stage3, self).__init__()
        ################################resnet101#######################################
        resnet_rgb = resnet34(pretrained=True)
        self.conv0 = nn.Sequential(nn.Conv2d(4, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=False))
        self.conv1 = nn.Sequential(nn.Conv2d(7, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=False))
        self.conv2_RGB = resnet_rgb.layer1  # 1/4, 256
        self.conv3_RGB = resnet_rgb.layer2  # 1/8, 512
        self.conv4_RGB = resnet_rgb.layer3  # 1/16, 1024
        self.conv5_RGB = resnet_rgb.layer4  # 1/32, 2048
        self.score = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1))


        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, RGB, FLOW, Static_sal, video_sal):
        input_static = torch.cat((RGB, Static_sal), 1)
        input_video = torch.cat((RGB, FLOW, video_sal), 1)
        x1 = self.conv0(input_static)
        x2 = self.conv1(input_video)

        e2_rgb = self.conv2_RGB(x1+x2)
        e3_rgb = self.conv3_RGB(e2_rgb)
        e4_rgb = self.conv4_RGB(e3_rgb)
        e5_rgb = self.conv5_RGB(e4_rgb)
        score = self.score(F.adaptive_avg_pool2d(e5_rgb,1))
        if self.training:
            return F.sigmoid(score)
        return F.sigmoid(score)
