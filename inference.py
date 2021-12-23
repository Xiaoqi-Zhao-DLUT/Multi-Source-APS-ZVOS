import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from utils.misc import check_mkdir,crf_refine
from models.stage1 import stage1
from models.stage2 import stage2
from models.stage3 import stage3
import numpy as np


torch.manual_seed(2018)
torch.cuda.set_device(0)

ckpt_path = './saved_model'
exp_name = 'prediction'
data_name = 'davis16'
args = {
    'crf_refine': True,
    'save_results': True
}

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

flow_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()


def main():
    model1 = stage1().cuda()
    model1.load_state_dict(torch.load('./saved_model/stage1/stage1.pth'))
    model1.eval()

    model2 = stage2().cuda()
    model2.load_state_dict(torch.load('./saved_model/stage2/stage2.pth'))
    model2.eval()

    model3 = stage3().cuda()
    model3.load_state_dict(torch.load('./saved_model/stage3/stage3.pth'))
    model3.eval()

    file_write_obj_img = "/val_image.txt"
    file_write_obj_flowmap = "/val_flow_RAFT.txt"
    img_list = []
    flow_list = []
    with open(os.path.join(file_write_obj_img), "r") as imgs:
        for img in imgs:
            _video = img.rstrip('\n')
            img_list.append(_video)
    with open(os.path.join(file_write_obj_flowmap), "r") as flows:
        for flow in flows:
            _video = flow.rstrip('\n')
            flow_list.append(_video)


    with torch.no_grad():
        for img_path, flow_path in zip(img_list, flow_list):
            split_path = img_path.split('/')
            img_name = split_path[-1][0:-4]
            video_name = split_path[-2]
            check_mkdir(os.path.join(ckpt_path, exp_name,data_name,video_name))
            img1 = Image.open(os.path.join(img_path)).convert('RGB')
            flow = Image.open(os.path.join(flow_path)).convert('RGB')
            img = img1
            w_,h_ = img1.size
            img1 = img1.resize([384,384],Image.BILINEAR)
            flow = flow.resize([384,384],Image.BILINEAR)
            img_var = Variable(img_transform(img1).unsqueeze(0), volatile=True).cuda()
            flow_var = Variable(flow_transform(flow).unsqueeze(0), volatile=True).cuda()
            e1_rgb, e2_rgb, e3_rgb, e4_rgb, e5_rgb, output4_d, output3_d, output2_d, output1_d, output4_sal, output3_sal, output2_sal, output1_sal = model1(
                img_var)
            prediction_static = output1_sal
            prediction_video = model2(e1_rgb, e2_rgb, e3_rgb, e4_rgb, e5_rgb, output4_d, output3_d, output2_d, output1_d,
                               output4_sal, output3_sal, output2_sal, output1_sal, flow_var)  # hed

            score = model3(img_var, flow_var, prediction_static, prediction_video)


            if score < 0.5:
                prediction = prediction_static
            else:
                prediction = prediction_video

            output_final = prediction.data.squeeze(0).cpu()
            prediction = to_pil(output_final)
            prediction = prediction.resize((w_, h_), Image.BILINEAR)
            if args['crf_refine']:
                prediction = crf_refine(np.array(img), np.array(prediction))
            prediction = np.array(prediction)
            prediction[prediction> 128] = 255
            prediction[prediction!= 255] = 0
            if args['save_results']:
                Image.fromarray(prediction).save(os.path.join(ckpt_path, exp_name,data_name,video_name,img_name + '.png'))



if __name__ == '__main__':
    main()
