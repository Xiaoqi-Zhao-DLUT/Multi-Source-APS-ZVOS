import os
import os.path
import torch.utils.data as data
from PIL import Image


def make_dataset(root):
    file_write_obj_img, file_write_obj_flowmap, file_write_obj_gt = root
    img_list = []
    flow_list = []
    gt_list = []
    with open(os.path.join(file_write_obj_img), "r") as imgs:
        # print(1)
        for img in imgs:
            _video = img.rstrip('\n')
            img_list.append(_video)
    with open(os.path.join(file_write_obj_flowmap), "r") as flows:
        for flow in flows:
            _video = flow.rstrip('\n')
            flow_list.append(_video)
    with open(os.path.join(file_write_obj_gt), "r") as gts:
        for gt in gts:
            _video = gt.rstrip('\n')
            gt_list.append(_video)
    img_list = sorted(img_list)
    flow_list = sorted(flow_list)
    gt_list = sorted(gt_list)
    return img_list, flow_list, gt_list


class ImageFolder(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs, self.flow, self.gt = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        flow_path = self.flow[index]
        gt_path = self.gt[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        flow = Image.open(flow_path).convert('RGB')
        if self.joint_transform is not None:
            img, flow, target = self.joint_transform(img, flow, target)
        if self.transform is not None:
            img = self.transform(img)
            flow = self.transform(flow)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, flow, target

    def __len__(self):
        return len(self.imgs)
