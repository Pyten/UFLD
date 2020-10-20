import numbers
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
#from config import cfg
import torch
import pdb
import cv2

# ===============================img tranforms============================

class Compose2(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, bbx=None):
        if bbx is None:
            for t in self.transforms:
                img, mask = t(img, mask)
            # Pyten-Debug
            # print(type(mask))
            return img, mask
        for t in self.transforms:
            img, mask, bbx = t(img, mask, bbx)
        return img, mask, bbx

class FreeScale(object):
    def __init__(self, size):
        self.size = size  # (h, w)

    def __call__(self, img, mask):
        return img.resize((self.size[1], self.size[0]), Image.BILINEAR), mask.resize((self.size[1], self.size[0]), Image.NEAREST)

class FreeScaleMask(object):
    def __init__(self,size):
        self.size = size
    def __call__(self,mask):
        return mask.resize((self.size[1], self.size[0]), Image.NEAREST)

class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        if img.size != mask.size:
            print(img.size)
            print(mask.size)
        assert img.size == mask.size
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img, mask
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)


class RandomRotate(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, angle):
        self.angle = angle
    # Pyten-20201009-ChangeforMultiLabel
    def __call__(self, image, labels):
        # Pyten-Debug
        # print(type(labels), len(labels), type(labels[0]))
        assert labels is None or image.size == labels[0].size

        angle = random.randint(0, self.angle * 2) - self.angle

        # Pyten-20201009-ChangeforMultiLabel
        # label = label.rotate(angle, resample=Image.NEAREST)
        # image = image.rotate(angle, resample=Image.BILINEAR)
        #  return image, label
        label_list = []
        if not isinstance(labels, list):
            labels = [labels]
        for label in labels:
            label = label.rotate(angle, resample=Image.NEAREST)
            label_list.append(label)
        image = image.rotate(angle, resample=Image.BILINEAR)
        if len(labels) < 2:
            return image, label_list[0]
        else:
            return image, label_list



# ===============================label tranforms============================

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def find_start_pos(row_sample,start_line):
    # row_sample = row_sample.sort()
    # for i,r in enumerate(row_sample):
    #     if r >= start_line:
    #         return i
    l,r = 0,len(row_sample)-1
    while True:
        mid = int((l+r)/2)
        if r - l == 1:
            return r
        if row_sample[mid] < start_line:
            l = mid
        if row_sample[mid] > start_line:
            r = mid
        if row_sample[mid] == start_line:
            return mid

class RandomLROffsetLABEL(object):
    def __init__(self,max_offset):
        self.max_offset = max_offset
    def __call__(self, img, labels):
        offset = np.random.randint(-self.max_offset,self.max_offset)
        w, h = img.size

        # Pyten-Debug
        # print("LROff", type(labels)) # len(labels)) # type(labels[0])
        img = np.array(img)
        if offset > 0:
            img[:,offset:,:] = img[:,0:w-offset,:]
            img[:,:offset,:] = 0
        if offset < 0:
            real_offset = -offset
            img[:,0:w-real_offset,:] = img[:,real_offset:,:]
            img[:,w-real_offset:,:] = 0

        # Pyten-20201009-Add-loop-for-seg-labels
        label_list = []
        if not isinstance(labels, list):
                    labels = [labels]
        
        for label in labels:

            label = np.array(label)
            if offset > 0:
                label[:,offset:] = label[:,0:w-offset]
                label[:,:offset] = 0
            if offset < 0:
                offset = -offset
                label[:,0:w-offset] = label[:,offset:]
                label[:,w-offset:] = 0
            
            label_list.append(label)

        if len(label_list) < 2:
            # Pyten-Debug
            # print("LR return no list")
            return Image.fromarray(img),Image.fromarray(label)
        else:
            # Pyten-Debug
            # print("LR return list")
            return Image.fromarray(img), [Image.fromarray(label) for label in label_list]

class RandomUDoffsetLABEL(object):
    def __init__(self,max_offset):
        self.max_offset = max_offset
    def __call__(self, img, labels):
        offset = np.random.randint(-self.max_offset,self.max_offset)
        w, h = img.size

        # Pyten-Debug
        # print("UDoff", type(labels), len(labels))
        img = np.array(img)
        if offset > 0:
            img[offset:,:,:] = img[0:h-offset,:,:]
            img[:offset,:,:] = 0
        if offset < 0:
            real_offset = -offset
            img[0:h-real_offset,:,:] = img[real_offset:,:,:]
            img[h-real_offset:,:,:] = 0

        # Pyten-20201009-Add-loop-for-seg-labels
        label_list = []
        if not isinstance(labels, list):
            # Pyten-Debug
            # print("not list")
            labels = [labels]
        for label in labels:

            label = np.array(label)
            if offset > 0:
                label[offset:,:] = label[0:h-offset,:]
                label[:offset,:] = 0
            if offset < 0:
                offset = -offset
                label[0:h-offset,:] = label[offset:,:]
                label[h-offset:,:] = 0

            label_list.append(label)

        if len(label_list) < 2:
            # Pyten-Debug
            # print("UD return no list")
            return Image.fromarray(img), Image.fromarray(label)
        else:
            # Pyten-Debug
            # print("UD return list")
            return Image.fromarray(img), [Image.fromarray(label) for label in label_list]

