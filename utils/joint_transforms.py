import numbers
import random

from PIL import Image, ImageOps
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, depth,mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, depth,mask = t(img, depth,mask)
        return img, depth,mask


class RandomCrop(object):
    def __init__(self, size,size1, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size1))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, depth,mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)
            depth = ImageOps.expand(depth, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img,depth, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR),depth.resize((tw, th), Image.NEAREST), mask.resize((tw, th), Image.NEAREST)
        return img.resize((tw, th), Image.BILINEAR), depth.resize((tw, th), Image.BILINEAR),mask.resize((tw, th), Image.NEAREST)


class RandomHorizontallyFlip(object):
    def __call__(self, img, depth,mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), depth.transpose(Image.FLIP_LEFT_RIGHT),mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, depth,mask


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, depth,mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR),depth.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)
