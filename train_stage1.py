import datetime
import os
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import joint_transforms
from utils.datasets import ImageFolder
from utils.misc import AvgMeter, check_mkdir
from utils.ssim_loss import SSIM
from models.stage1 import stage1
from torch.backends import cudnn
import torch.nn.functional as functional


cudnn.benchmark = True

torch.manual_seed(2018)
torch.cuda.set_device(0)

##########################hyperparameters###############################
ckpt_path = './saved_model'
exp_name = 'stage1'
args = {
    'iter_num':76425,
    'train_batch_size': 4,
    'last_iter': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'snapshot': ''
}
##########################data augmentation###############################
joint_transform = joint_transforms.Compose([
    joint_transforms.RandomCrop(384,384),
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(10)
])
img_transform = transforms.Compose([
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
target_transform = transforms.ToTensor()
##########################################################################
train_data = os.path.join('/RGBDSOD/train_data')
train_set = ImageFolder(train_data, joint_transform, img_transform, target_transform)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=12, shuffle=True)
criterion = nn.BCEWithLogitsLoss().cuda()
criterion_BCE = nn.BCELoss().cuda()
criterion_MAE = nn.L1Loss().cuda()
criterion_MSE = nn.MSELoss().cuda()
criterion_smoothl1 = nn.SmoothL1Loss().cuda()
criterion_ssim = SSIM(window_size=11,size_average=True)
criterion_triplet_loss = nn.TripletMarginLoss(margin=1.5, p=2)
criterion_triplet_loss_e2 = nn.TripletMarginLoss(margin=1, p=2)
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')


def ssimmae(pre,gt):
    maeloss = criterion_MAE(pre,gt)
    ssimloss = 1-criterion_ssim(pre,gt)
    loss = ssimloss+maeloss
    return loss


def main():

    model = stage1()
    net = model.cuda().train()
    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])
    if len(args['snapshot']) > 0:
        print ('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']
    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net,optimizer)


def train(net, optimizer):
    curr_iter = args['last_iter']
    while True:
        total_loss_record, loss1_record, loss2_record,loss3_record,loss4_record,loss5_record,loss6_record,loss7_record,loss8_record,loss9_record ,loss10_record,loss11_record ,loss12_record ,loss13_record ,loss14_record ,loss15_record ,loss16_record,loss17_record ,loss18_record,loss19_record, loss20_record,loss21_record,loss22_record,loss23_record,loss24_record,loss25_record,loss26_record,loss27_record ,loss28_record,loss29_record ,loss30_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(),AvgMeter(),AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(),AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(),AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']

            inputs, depth, labels = data
            labels[labels > 0.5] = 1
            labels[labels != 1] = 0
            batch_size = inputs.size(0)
            h = inputs.size(2)
            inputs = Variable(inputs).cuda()
            depth = Variable(depth).cuda()
            labels = Variable(labels).cuda()
            P5_D,P4_D,P3_D,P2_D,P1_D,P5_sal,P4_sal,P3_sal,P2_sal,P1_sal, = net(inputs)


            depth6 = functional.interpolate(depth, size=h//32, mode='bilinear')
            depth5 = functional.interpolate(depth, size=h//16, mode='bilinear')
            depth4 = functional.interpolate(depth, size=h//8, mode='bilinear')
            depth3 = functional.interpolate(depth, size=h//4, mode='bilinear')
            depth2 = functional.interpolate(depth, size=h//2, mode='bilinear')

            labels6 = functional.interpolate(labels, size=h//32, mode='bilinear')
            labels5 = functional.interpolate(labels, size=h//16, mode='bilinear')
            labels4 = functional.interpolate(labels, size=h//8, mode='bilinear')
            labels3 = functional.interpolate(labels, size=h//4, mode='bilinear')
            labels2 = functional.interpolate(labels, size=h//2, mode='bilinear')
            optimizer.zero_grad()


            loss1 = ssimmae(P1_D,depth)
            loss2 = ssimmae(P2_D,depth3)
            loss3 = ssimmae(P3_D,depth4)
            loss4 = ssimmae(P4_D,depth5)
            loss5 = ssimmae(P5_D,depth6)
            loss6 = criterion_BCE(P1_sal,labels)
            loss7 = criterion_BCE(P2_sal,labels3)
            loss8 = criterion_BCE(P3_sal,labels4)
            loss9 = criterion_BCE(P4_sal,labels5)
            loss10 = criterion_BCE(P5_sal,labels6)

            total_loss = loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9+loss10
            total_loss.backward()
            optimizer.step()
            total_loss_record.update(total_loss.item(), batch_size)
            loss1_record.update(loss1.item(), batch_size)
            loss2_record.update(loss2.item(), batch_size)
            loss3_record.update(loss3.item(), batch_size)
            loss4_record.update(loss4.item(), batch_size)
            loss5_record.update(loss5.item(), batch_size)
            loss6_record.update(loss6.item(), batch_size)
            loss7_record.update(loss7.item(), batch_size)
            loss8_record.update(loss8.item(), batch_size)
            loss9_record.update(loss9.item(), batch_size)
            loss10_record.update(loss10.item(), batch_size)

            curr_iter += 1

            log = '[iter %d], [total loss %.5f],[loss1 %.5f],[loss6 %.5f],[lr %.13f] '  % \
                     (curr_iter, total_loss_record.avg, loss1_record.avg,  loss6_record.avg,optimizer.param_groups[1]['lr'])
            print(log)
            open(log_path, 'a').write(log + '\n')
            if curr_iter == args['iter_num']:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                return
            #############end###############

if __name__ == '__main__':
    main()
