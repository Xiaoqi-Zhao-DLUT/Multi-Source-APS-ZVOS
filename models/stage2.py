import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import resnet101


class stage2(nn.Module):

    def __init__(self):
        super(stage2, self).__init__()
        ################################resnet101#######################################
        resnet_flow = resnet101(pretrained=True)
        self.conv1_flow = resnet_flow.conv1
        self.bn1_flow = resnet_flow.bn1
        self.relu_flow = resnet_flow.relu  # 1/2, 64
        self.maxpool_flow = resnet_flow.maxpool
        self.conv2_flow = resnet_flow.layer1  # 1/4, 256
        self.conv3_flow = resnet_flow.layer2  # 1/8, 512
        self.conv4_flow = resnet_flow.layer3  # 1/16, 1024
        self.conv5_flow = resnet_flow.layer4  # 1/32, 2048
        ###############################Transition Layer########################################
        self.T5_flow = nn.Sequential(nn.Conv2d(2048, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU())
        self.T4_flow = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU())
        self.T3_flow = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU())
        self.T2_flow = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU())
        ###############################Transition Layer########################################
        self.T5_rgb = nn.Sequential(nn.Conv2d(2048, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU())
        self.T4_rgb = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU())
        self.T3_rgb = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU())
        self.T2_rgb = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU())

        self.T5_fuse_initial = nn.Sequential(nn.Conv2d(256*2, 256*2, kernel_size=1), nn.BatchNorm2d(256*2), nn.PReLU(),nn.Conv2d(256*2, 256*2, kernel_size=1), nn.BatchNorm2d(256*2), nn.PReLU())
        self.T5_rgb_sa = nn.Conv2d(256*2, 1, kernel_size=3, padding=1)
        self.T5_rgb_PPM_fuse = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.T5_flow_sa = nn.Conv2d(256 * 2, 1, kernel_size=3, padding=1)
        self.T5_flow_PPM_fuse = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.T5_rgb_sa_fuse = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.T5_flow_sa_fuse = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.T5_fuse_positive = nn.Sequential(nn.Conv2d(256*2, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())
        self.T5_fuse_negative = nn.Sequential(nn.Conv2d(256*2, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())
        self.T5_fuse = nn.Sequential(nn.Conv2d(256*2, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())

        self.T4_fuse_initial = nn.Sequential(nn.Conv2d(256 * 4, 256 * 4, kernel_size=1), nn.BatchNorm2d(256 * 4),
                                            nn.PReLU(), nn.Conv2d(256 * 4, 256 * 4, kernel_size=1),
                                            nn.BatchNorm2d(256 * 4), nn.PReLU())
        self.T4_rgb_sa = nn.Conv2d(256 * 4, 1, kernel_size=3, padding=1)
        self.T4_rgb_PPM_fuse = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.T4_flow_sa = nn.Conv2d(256 * 4, 1, kernel_size=3, padding=1)
        self.T4_flow_PPM_fuse = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.T4_depth_sa = nn.Conv2d(256 * 4, 1, kernel_size=3, padding=1)
        self.T4_depth_PPM_fuse = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.T4_sal_sa = nn.Conv2d(256 * 4, 1, kernel_size=3, padding=1)
        self.T4_sal_PPM_fuse = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.T4_rgb_sa_fuse = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
                                            nn.PReLU())
        self.T4_flow_sa_fuse = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
                                             nn.PReLU())
        self.T4_depth_sa_fuse = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
                                             nn.PReLU())
        self.T4_sal_sa_fuse = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
                                             nn.PReLU())
        self.T4_fuse_positive = nn.Sequential(nn.Conv2d(256*4, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())
        self.T4_fuse_negative = nn.Sequential(nn.Conv2d(256*4, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())
        self.T4_fuse = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())

        self.T3_fuse_initial = nn.Sequential(nn.Conv2d(256 * 4, 256 * 4, kernel_size=1), nn.BatchNorm2d(256 * 4),
                                            nn.PReLU(), nn.Conv2d(256 * 4, 256 * 4, kernel_size=1),
                                            nn.BatchNorm2d(256 * 4), nn.PReLU())
        self.T3_rgb_sa = nn.Conv2d(256 * 4, 1, kernel_size=3, padding=1)
        self.T3_rgb_PPM_fuse = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.T3_flow_sa = nn.Conv2d(256 * 4, 1, kernel_size=3, padding=1)
        self.T3_flow_PPM_fuse = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.T3_depth_sa = nn.Conv2d(256 * 4, 1, kernel_size=3, padding=1)
        self.T3_depth_PPM_fuse = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.T3_sal_sa = nn.Conv2d(256 * 4, 1, kernel_size=3, padding=1)
        self.T3_sal_PPM_fuse = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.T3_rgb_sa_fuse = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
                                            nn.PReLU())
        self.T3_flow_sa_fuse = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
                                             nn.PReLU())
        self.T3_depth_sa_fuse = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
                                              nn.PReLU())
        self.T3_sal_sa_fuse = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
                                            nn.PReLU())
        self.T3_fuse_positive = nn.Sequential(nn.Conv2d(256 * 4, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())
        self.T3_fuse_negative = nn.Sequential(nn.Conv2d(256 * 4, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())
        self.T3_fuse = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())

        self.T2_fuse_initial = nn.Sequential(nn.Conv2d(256 * 2+64*2, 256 * 2+64*2, kernel_size=1), nn.BatchNorm2d(256 * 2+64*2),
                                            nn.PReLU(), nn.Conv2d(256 * 2+64*2, 256 * 2+64*2, kernel_size=1),
                                            nn.BatchNorm2d(256 * 2+64*2), nn.PReLU())
        self.T2_rgb_sa = nn.Conv2d(256 * 2+64*2, 1, kernel_size=3, padding=1)
        self.T2_rgb_PPM_fuse = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.T2_flow_sa = nn.Conv2d(256 * 2+64*2, 1, kernel_size=3, padding=1)
        self.T2_flow_PPM_fuse = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.T2_depth_sa = nn.Conv2d(256 * 2+64*2, 1, kernel_size=3, padding=1)
        self.T2_depth_PPM_fuse = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.T2_sal_sa = nn.Conv2d(256 * 2+64*2, 1, kernel_size=3, padding=1)
        self.T2_sal_PPM_fuse = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.T2_rgb_sa_fuse = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
                                            nn.PReLU())
        self.T2_flow_sa_fuse = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
                                             nn.PReLU())
        self.T2_depth_sa_fuse = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                              nn.PReLU())
        self.T2_sal_sa_fuse = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                            nn.PReLU())
        self.T2_fuse_positive = nn.Sequential(nn.Conv2d(256*2+64*2, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.T2_fuse_negative = nn.Sequential(nn.Conv2d(256*2+64*2, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.T2_fuse = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())


        self.T1_fuse_initial = nn.Sequential(nn.Conv2d(64*2, 64*2, kernel_size=1), nn.BatchNorm2d(64*2), nn.PReLU(),nn.Conv2d(64*2, 64*2, kernel_size=1), nn.BatchNorm2d(64*2), nn.PReLU())
        self.T1_rgb_sa = nn.Conv2d(64*2, 1, kernel_size=3, padding=1)
        self.T1_rgb_PPM_fuse = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.T1_flow_sa = nn.Conv2d(64 * 2, 1, kernel_size=3, padding=1)
        self.T1_flow_PPM_fuse = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.T1_rgb_sa_fuse = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.T1_flow_sa_fuse = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.T1_fuse_positive = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.T1_fuse_negative = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.T1_fuse = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())


        self.output4_sal = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512),
                                           nn.PReLU())
        self.output3_sal = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
                                           nn.PReLU())
        self.output2_sal = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.output1_sal = nn.Sequential(nn.Conv2d(128, 1, kernel_size=3, padding=1))

        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, e1_rgb, e2_rgb, e3_rgb, e4_rgb, e5_rgb, output4_d, output3_d, output2_d, output1_d, output4_sal, output3_sal, output2_sal, output1_sal, flows):
        input = flows
        B, _, _, _ = input.size()
        x = self.conv1_flow(flows)
        x = self.bn1_flow(x)
        e1_flow = self.relu_flow(x)  # 1/2, 64
        x = self.maxpool_flow(e1_flow)  # 1/4, 64
        e2_flow = self.conv2_flow(x)
        e3_flow = self.conv3_flow(e2_flow)
        e4_flow = self.conv4_flow(e3_flow)
        e5_flow = self.conv5_flow(e4_flow)
        # print(e1_rgb.shape,e2_rgb.shape,e3_rgb.shape,e4_rgb.shape,e5_rgb.shape)
        T5_flow = self.T5_flow(e5_flow)
        T4_flow = self.T4_flow(e4_flow)
        T3_flow = self.T3_flow(e3_flow)
        T2_flow = self.T2_flow(e2_flow)
        T1_flow = e1_flow

        T5_rgb = self.T5_rgb(e5_rgb)
        T4_rgb = self.T4_rgb(e4_rgb)
        T3_rgb = self.T3_rgb(e3_rgb)
        T2_rgb = self.T2_rgb(e2_rgb)
        T1_rgb = e1_rgb

        T5_fuse_initial = self.T5_fuse_initial(torch.cat((T5_rgb,T5_flow),1))
        T5_rgb_sa = F.sigmoid(self.T5_rgb_PPM_fuse(self.PPM(self.T5_rgb_sa(T5_fuse_initial))))
        T5_rgb_sa_enhanced = self.T5_rgb_sa_fuse(T5_rgb*T5_rgb_sa)+T5_rgb
        T5_flow_sa = F.sigmoid(self.T5_flow_PPM_fuse(self.PPM(self.T5_flow_sa(T5_fuse_initial))))
        T5_flow_sa_enhanced = self.T5_flow_sa_fuse(T5_flow * T5_flow_sa) + T5_flow
        T5_fuse_positive = self.T5_fuse_positive(torch.cat((T5_rgb_sa_enhanced, T5_flow_sa_enhanced), 1))
        T5_fuse_negative = self.T5_fuse_negative(torch.cat((T5_rgb_sa_enhanced, T5_flow_sa_enhanced), 1))
        T5_fuse = self.T5_fuse(T5_fuse_positive - T5_fuse_negative)

        T4_fuse_initial = self.T4_fuse_initial(torch.cat((T4_rgb,T4_flow,output4_d,output4_sal),1))
        T4_rgb_sa = F.sigmoid(self.T4_rgb_PPM_fuse(self.PPM(self.T4_rgb_sa(T4_fuse_initial))))
        T4_rgb_sa_enhanced = self.T4_rgb_sa_fuse(T4_rgb * T4_rgb_sa) + T4_rgb
        T4_flow_sa = F.sigmoid(self.T4_flow_PPM_fuse(self.PPM(self.T4_flow_sa(T4_fuse_initial))))
        T4_flow_sa_enhanced = self.T4_flow_sa_fuse(T4_flow * T4_flow_sa) + T4_flow
        T4_depth_sa = F.sigmoid(self.T4_depth_PPM_fuse(self.PPM(self.T4_depth_sa(T4_fuse_initial))))
        T4_depth_sa_enhanced = self.T4_depth_sa_fuse(output4_d * T4_depth_sa) + output4_d
        T4_sal_sa = F.sigmoid(self.T4_sal_PPM_fuse(self.PPM(self.T4_sal_sa(T4_fuse_initial))))
        T4_sal_sa_enhanced = self.T4_sal_sa_fuse(output4_sal * T4_sal_sa) + output4_sal
        T4_fuse_positive = self.T4_fuse_positive(torch.cat((T4_rgb_sa_enhanced, T4_flow_sa_enhanced,T4_depth_sa_enhanced,T4_sal_sa_enhanced), 1))
        T4_fuse_negative = self.T4_fuse_negative(torch.cat((T4_rgb_sa_enhanced, T4_flow_sa_enhanced,T4_depth_sa_enhanced,T4_sal_sa_enhanced), 1))
        T4_fuse = self.T4_fuse(T4_fuse_positive-T4_fuse_negative)

        T3_fuse_initial = self.T3_fuse_initial(torch.cat((T3_rgb, T3_flow, output3_d, output3_sal), 1))
        T3_rgb_sa = F.sigmoid(self.T3_rgb_PPM_fuse(self.PPM(self.T3_rgb_sa(T3_fuse_initial))))
        T3_rgb_sa_enhanced = self.T3_rgb_sa_fuse(T3_rgb * T3_rgb_sa) + T3_rgb
        T3_flow_sa = F.sigmoid(self.T3_flow_PPM_fuse(self.PPM(self.T3_flow_sa(T3_fuse_initial))))
        T3_flow_sa_enhanced = self.T3_flow_sa_fuse(T3_flow * T3_flow_sa) + T3_flow
        T3_depth_sa = F.sigmoid(self.T3_depth_PPM_fuse(self.PPM(self.T3_depth_sa(T3_fuse_initial))))
        T3_depth_sa_enhanced = self.T3_depth_sa_fuse(output3_d * T3_depth_sa) + output3_d
        T3_sal_sa = F.sigmoid(self.T3_sal_PPM_fuse(self.PPM(self.T3_sal_sa(T3_fuse_initial))))
        T3_sal_sa_enhanced = self.T3_sal_sa_fuse(output3_sal * T3_sal_sa) + output3_sal
        T3_fuse_positive = self.T3_fuse_positive(torch.cat((T3_rgb_sa_enhanced, T3_flow_sa_enhanced, T3_depth_sa_enhanced, T3_sal_sa_enhanced), 1))
        T3_fuse_negative = self.T3_fuse_negative(torch.cat((T3_rgb_sa_enhanced, T3_flow_sa_enhanced, T3_depth_sa_enhanced, T3_sal_sa_enhanced), 1))
        T3_fuse = self.T3_fuse(T3_fuse_positive-T3_fuse_negative)

        T2_fuse_initial = self.T2_fuse_initial(torch.cat((T2_rgb, T2_flow, output2_d, output2_sal), 1))
        T2_rgb_sa = F.sigmoid(self.T2_rgb_PPM_fuse(self.PPM(self.T2_rgb_sa(T2_fuse_initial))))
        T2_rgb_sa_enhanced = self.T2_rgb_sa_fuse(T2_rgb * T2_rgb_sa) + T2_rgb
        T2_flow_sa = F.sigmoid(self.T2_flow_PPM_fuse(self.PPM(self.T2_flow_sa(T2_fuse_initial))))
        T2_flow_sa_enhanced = self.T2_flow_sa_fuse(T2_flow * T2_flow_sa) + T2_flow
        T2_depth_sa = F.sigmoid(self.T2_depth_PPM_fuse(self.PPM(self.T2_depth_sa(T2_fuse_initial))))
        T2_depth_sa_enhanced = self.T2_depth_sa_fuse(output2_d * T2_depth_sa) + output2_d
        T2_sal_sa = F.sigmoid(self.T2_sal_PPM_fuse(self.PPM(self.T2_sal_sa(T2_fuse_initial))))
        T2_sal_sa_enhanced = self.T2_sal_sa_fuse(output2_sal * T2_sal_sa) + output2_sal
        T2_fuse_positive = self.T2_fuse_positive(torch.cat((T2_rgb_sa_enhanced, T2_flow_sa_enhanced, T2_depth_sa_enhanced, T2_sal_sa_enhanced), 1))
        T2_fuse_negative = self.T2_fuse_negative(torch.cat((T2_rgb_sa_enhanced, T2_flow_sa_enhanced, T2_depth_sa_enhanced, T2_sal_sa_enhanced), 1))
        T2_fuse = self.T2_fuse(T2_fuse_positive-T2_fuse_negative)

        T1_fuse_initial = self.T1_fuse_initial(torch.cat((T1_rgb, T1_flow), 1))
        T1_rgb_sa = F.sigmoid(self.T1_rgb_PPM_fuse(self.PPM(self.T1_rgb_sa(T1_fuse_initial))))
        T1_rgb_sa_enhanced = self.T1_rgb_sa_fuse(T1_rgb * T1_rgb_sa) + T1_rgb
        T1_flow_sa = F.sigmoid(self.T1_flow_PPM_fuse(self.PPM(self.T1_flow_sa(T1_fuse_initial))))
        T1_flow_sa_enhanced = self.T1_flow_sa_fuse(T1_flow * T1_flow_sa) + T1_flow
        T1_fuse_positive = self.T1_fuse_positive(torch.cat((T1_rgb_sa_enhanced, T1_flow_sa_enhanced), 1))
        T1_fuse_negative = self.T1_fuse_negative(torch.cat((T1_rgb_sa_enhanced, T1_flow_sa_enhanced), 1))
        T1_fuse = self.T1_fuse(T1_fuse_positive-T1_fuse_negative)

        ################################decoder#######################################
        output4_sal = self.output4_sal(F.upsample(T5_fuse, size=e4_rgb.size()[2:], mode='bilinear') + T4_fuse)
        output3_sal = self.output3_sal(F.upsample(output4_sal, size=e3_rgb.size()[2:], mode='bilinear') + T3_fuse)
        output2_sal = self.output2_sal(F.upsample(output3_sal, size=e2_rgb.size()[2:], mode='bilinear') + T2_fuse)
        output1_sal = self.output1_sal(F.upsample(output2_sal, size=e1_rgb.size()[2:], mode='bilinear') + T1_fuse)
        output1_sal = F.upsample(output1_sal, size=input.size()[2:], mode='bilinear')
        if self.training:
            return output1_sal
        return F.sigmoid(output1_sal)


    def PPM(self,Pool_F):

        Pool_F2 = F.avg_pool2d(Pool_F,kernel_size=(2,2))
        Pool_F4 = F.avg_pool2d(Pool_F,kernel_size=(4,4))
        Pool_F6 = F.avg_pool2d(Pool_F,kernel_size=(6,6))
        Pool_Fgolobal = F.adaptive_avg_pool2d(Pool_F,1)
        fuse = torch.cat((F.upsample(Pool_F2, size=Pool_F.size()[2:], mode='bilinear'), F.upsample(Pool_F4, size=Pool_F.size()[2:], mode='bilinear'),F.upsample(Pool_F6, size=Pool_F.size()[2:], mode='bilinear'),F.upsample(Pool_Fgolobal, size=Pool_F.size()[2:], mode='bilinear')),1)
        return fuse


