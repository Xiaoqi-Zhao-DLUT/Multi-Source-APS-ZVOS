import os
import os.path
import torch.utils.data as data
from PIL import Image

def make_dataset(root):
    img_path = os.path.join(root, 'RGB')
    depth_path = os.path.join(root, 'depth')
    gt_path = os.path.join(root, 'GT')
    img_list = [os.path.splitext(f)[0]
                 for f in os.listdir(gt_path) if f.endswith('.png')]
    #######################################################################
    return [(os.path.join(img_path, img_name + '.jpg'),
             os.path.join(depth_path,img_name + '.png'),os.path.join(gt_path, img_name + '.png')) for img_name in img_list]

class ImageFolder(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        img_path, depth_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        depth = Image.open(depth_path).convert('L')
        if self.joint_transform is not None:
            img, depth, target = self.joint_transform(img,depth, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            depth = self.target_transform(depth)

        return img, depth,target

    def __len__(self):
        return len(self.imgs)
