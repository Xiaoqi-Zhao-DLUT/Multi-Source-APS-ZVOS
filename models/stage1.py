import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import resnet101



class stage1(nn.Module):

    def __init__(self):
        super(stage1, self).__init__()
        ################################resnet101#######################################
        resnet_rgb = resnet101(pretrained=True)
        self.conv1_RGB = resnet_rgb.conv1
        self.bn1_RGB = resnet_rgb.bn1
        self.relu_RGB = resnet_rgb.relu  # 1/2, 64
        self.maxpool_RGB = resnet_rgb.maxpool
        self.conv2_RGB = resnet_rgb.layer1  # 1/4, 256
        self.conv3_RGB = resnet_rgb.layer2  # 1/8, 512
        self.conv4_RGB = resnet_rgb.layer3  # 1/16, 1024
        self.conv5_RGB = resnet_rgb.layer4  # 1/32, 2048

        ###############################Transition Layer########################################
        self.T5_depth = nn.Sequential(nn.Conv2d(2048, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU())
        self.T4_depth = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU())
        self.T3_depth = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU())
        self.T2_depth = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU())
        self.T1_depth = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU())
        ###############################Transition Layer########################################
        self.T5_sal = nn.Sequential(nn.Conv2d(2048, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU())
        self.T4_sal = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU())
        self.T3_sal = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU())
        self.T2_sal = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU())
        self.T1_sal = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU())

        self.output4_depth = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.output3_depth = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.output2_depth = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.PReLU())
        self.output1_depth = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))


        self.output4_sal = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
                                           nn.PReLU())
        self.output3_sal = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
                                           nn.PReLU())
        self.output2_sal = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.output1_sal = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))


        self.sideout5_depth = nn.Sequential(nn.Conv2d(256, 1, kernel_size=3, padding=1))
        self.sideout4_depth = nn.Sequential(nn.Conv2d(256, 1, kernel_size=3, padding=1))
        self.sideout3_depth = nn.Sequential(nn.Conv2d(256, 1, kernel_size=3, padding=1))
        self.sideout2_depth = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))


        self.sideout5_sal =  nn.Sequential(nn.Conv2d(256, 1, kernel_size=3, padding=1))
        self.sideout4_sal =  nn.Sequential(nn.Conv2d(256, 1, kernel_size=3, padding=1))
        self.sideout3_sal =  nn.Sequential(nn.Conv2d(256, 1, kernel_size=3, padding=1))
        self.sideout2_sal =  nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))

        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, x):

        input = x
        B, _, _, _ = input.size()
        x = self.conv1_RGB(x)
        x = self.bn1_RGB(x)
        e1_rgb = self.relu_RGB(x)
        x = self.maxpool_RGB(e1_rgb)
        e2_rgb = self.conv2_RGB(x)
        e3_rgb = self.conv3_RGB(e2_rgb)
        e4_rgb = self.conv4_RGB(e3_rgb)
        e5_rgb = self.conv5_RGB(e4_rgb)


        T5_d = self.T5_depth(e5_rgb)
        T4_d = self.T4_depth(e4_rgb)
        T3_d = self.T3_depth(e3_rgb)
        T2_d = self.T2_depth(e2_rgb)
        T1_d = self.T1_depth(e1_rgb)
        T5_sal = self.T5_sal(e5_rgb)
        T4_sal = self.T4_sal(e4_rgb)
        T3_sal = self.T3_sal(e3_rgb)
        T2_sal = self.T2_sal(e2_rgb)
        T1_sal = self.T1_sal(e1_rgb)

        ################################decoder#######################################
        sideout5_d = self.sideout5_depth(T5_d)
        output4_d = self.output4_depth(F.upsample(T5_d, size=e4_rgb.size()[2:], mode='bilinear')+T4_d)
        sideout4_d = self.sideout4_depth(output4_d)
        output3_d = self.output3_depth(F.upsample(output4_d, size=e3_rgb.size()[2:], mode='bilinear')+T3_d)
        sideout3_d = self.sideout3_depth(output3_d)
        output2_d = self.output2_depth(F.upsample(output3_d, size=e2_rgb.size()[2:], mode='bilinear')+T2_d)
        sideout2_d = self.sideout2_depth(output2_d)
        output1_d = self.output1_depth(F.upsample(output2_d, size=e1_rgb.size()[2:], mode='bilinear')+T1_d)
        output1_d = F.upsample(output1_d, size=input.size()[2:], mode='bilinear')


        sideout5_sal = self.sideout5_sal(T5_sal)
        output4_sal = self.output4_sal(F.upsample(T5_sal, size=e4_rgb.size()[2:], mode='bilinear') + T4_sal)
        sideout4_sal = self.sideout4_sal(output4_sal)
        output3_sal = self.output3_sal(F.upsample(output4_sal, size=e3_rgb.size()[2:], mode='bilinear') + T3_sal)
        sideout3_sal = self.sideout3_sal(output3_sal)
        output2_sal = self.output2_sal(F.upsample(output3_sal, size=e2_rgb.size()[2:], mode='bilinear') + T2_sal)
        sideout2_sal = self.sideout2_sal(output2_sal)
        output1_sal = self.output1_sal(F.upsample(output2_sal, size=e1_rgb.size()[2:], mode='bilinear') + T1_sal)
        output1_sal = F.upsample(output1_sal, size=input.size()[2:], mode='bilinear')
        if self.training:
            return F.sigmoid(sideout5_d),F.sigmoid(sideout4_d),F.sigmoid(sideout3_d),F.sigmoid(sideout2_d),F.sigmoid(output1_d),F.sigmoid(sideout5_sal),F.sigmoid(sideout4_sal),F.sigmoid(sideout3_sal),F.sigmoid(sideout2_sal),F.sigmoid(output1_sal)
        return e1_rgb,e2_rgb,e3_rgb,e4_rgb,e5_rgb, output4_d, output3_d,output2_d, F.sigmoid(output1_d), output4_sal, output3_sal, output2_sal, F.sigmoid(output1_sal)




if __name__ == "__main__":
    model = stage1()
    input = torch.autograd.Variable(torch.randn(4, 3, 384, 384))
    output = model(input)
