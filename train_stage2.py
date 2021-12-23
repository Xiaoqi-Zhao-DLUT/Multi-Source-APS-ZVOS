import datetime
import os
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import joint_transforms
from utils.datasets_video import ImageFolder
from utils.misc import AvgMeter, check_mkdir
from utils.ssim_loss import SSIM
from models.stage1 import stage1
from models.stage2 import stage2
from torch.backends import cudnn
import torch.nn.functional as functional


cudnn.benchmark = True

torch.manual_seed(2018)
torch.cuda.set_device(0)

##########################hyperparameters###############################
ckpt_path = './saved_model'
exp_name = 'stage2'
args = {
    'iter_num': 51200,
    'train_batch_size': 4,
    'last_iter': 0,
    'lr': 5e-3,
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
file_write_obj_img = "train_img.txt"
file_write_obj_flowmap = "train_flow_raft.txt"
file_write_obj_gt = "train_gt.txt"

train_set = ImageFolder([file_write_obj_img,file_write_obj_flowmap,file_write_obj_gt], joint_transform, img_transform, target_transform)
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
    model1 = stage1().cuda()
    model1.load_state_dict(torch.load('./saved_model/stage1/stage1.pth'))
    model1.eval()
    for param in model1.parameters():
        param.requires_grad = False

    model = stage2()
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
    train(net, model1, optimizer)


def train(net, model1, optimizer):
    curr_iter = args['last_iter']
    while True:
        total_loss_record, loss1_record, loss2_record,loss3_record,loss4_record,loss5_record,loss6_record,loss7_record,loss8_record,loss9_record ,loss10_record,loss11_record ,loss12_record ,loss13_record ,loss14_record ,loss15_record ,loss16_record,loss17_record ,loss18_record,loss19_record, loss20_record,loss21_record,loss22_record,loss23_record,loss24_record,loss25_record,loss26_record,loss27_record ,loss28_record,loss29_record ,loss30_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(),AvgMeter(),AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(),AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(),AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']

            inputs, flows, labels = data
            labels[labels > 0.5] = 1
            labels[labels != 1] = 0
            batch_size = inputs.size(0)
            h = inputs.size(2)
            inputs = Variable(inputs).cuda()
            flows = Variable(flows).cuda()
            labels = Variable(labels).cuda()
            e1_rgb, e2_rgb, e3_rgb, e4_rgb, e5_rgb, output4_d, output3_d, output2_d, output1_d, output4_sal, output3_sal, output2_sal, output1_sal = model1(inputs)
            output1_videosal = net(e1_rgb, e2_rgb, e3_rgb, e4_rgb, e5_rgb, output4_d, output3_d, output2_d, output1_d, output4_sal, output3_sal, output2_sal, output1_sal, flows) #hed

            labels6 = functional.interpolate(labels, size=h//32, mode='bilinear')
            labels5 = functional.interpolate(labels, size=h//16, mode='bilinear')
            labels4 = functional.interpolate(labels, size=h//8, mode='bilinear')
            labels3 = functional.interpolate(labels, size=h//4, mode='bilinear')
            labels2 = functional.interpolate(labels, size=h//2, mode='bilinear')
            optimizer.zero_grad()

            loss1 = criterion(output1_videosal,labels)


            total_loss = loss1
            total_loss.backward()
            optimizer.step()
            total_loss_record.update(total_loss.item(), batch_size)
            loss1_record.update(loss1.item(), batch_size)
            curr_iter += 1

            log = '[iter %d], [total loss %.5f],[loss1 %.5f],[lr %.13f] '  % \
                     (curr_iter, total_loss_record.avg, loss1_record.avg, optimizer.param_groups[1]['lr'])
            print(log)
            open(log_path, 'a').write(log + '\n')

            if curr_iter %2560 == 0:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))

            if curr_iter == args['iter_num']:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                return
            #############end###############

if __name__ == '__main__':
    main()
