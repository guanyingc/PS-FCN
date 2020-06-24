from __future__ import division
import os
import numpy as np
#from scipy.ndimage import imread
from imageio import imread

import torch
import torch.utils.data as data

from datasets import pms_transforms
from . import util
np.random.seed(0)

class PS_Synth_Dataset(data.Dataset):
    def __init__(self, args, root, split='train'):
        self.root   = os.path.join(root)
        self.split  = split
        self.args   = args
        self.shape_list = util.readList(os.path.join(self.root, split + '_mtrl.txt'))

    def _getInputPath(self, index):
        shape, mtrl = self.shape_list[index].split('/')
        normal_path = os.path.join(self.root, 'Images', shape, shape + '_normal.png')
        img_dir     = os.path.join(self.root, 'Images', self.shape_list[index])
        img_list    = util.readList(os.path.join(img_dir, '%s_%s.txt' % (shape, mtrl)))

        data = np.genfromtxt(img_list, dtype='str', delimiter=' ')
        select_idx = np.random.permutation(data.shape[0])[:self.args.in_img_num]
        idxs = ['%03d' % (idx) for idx in select_idx]
        data   = data[select_idx, :]
        imgs   = [os.path.join(img_dir, img) for img in data[:, 0]]
        lights = data[:, 1:4].astype(np.float32)
        return normal_path, imgs, lights

    def __getitem__(self, index):
        normal_path, img_list, lights = self._getInputPath(index)
        normal = imread(normal_path).astype(np.float32) / 255.0 * 2 - 1
        imgs   =  []
        for i in img_list:
            img = imread(i).astype(np.float32) / 255.0
            imgs.append(img)
        img = np.concatenate(imgs, 2)

        h, w, c = img.shape
        crop_h, crop_w = self.args.crop_h, self.args.crop_w
        if self.args.rescale:
            sc_h = np.random.randint(crop_h, h)
            sc_w = np.random.randint(crop_w, w)
            img, normal = pms_transforms.rescale(img, normal, [sc_h, sc_w])

        if self.args.crop:
            img, normal = pms_transforms.randomCrop(img, normal, [crop_h, crop_w])

        if self.args.color_aug and not self.args.normalize:
            img = (img * np.random.uniform(1, 3)).clip(0, 2)

        if self.args.normalize:
            imgs = np.split(img, img.shape[2]//3, 2)
            imgs = pms_transforms.normalize(imgs)
            img = np.concatenate(imgs, 2)

        if self.args.noise_aug:
            img = pms_transforms.randomNoiseAug(img, self.args.noise)

        mask   = pms_transforms.normalToMask(normal)
        normal = normal * mask.repeat(3, 2)
        norm  = np.sqrt((normal * normal).sum(2, keepdims=True))
        normal = normal / (norm + 1e-10)

        item = {'N': normal, 'img': img, 'mask': mask}
        for k in item.keys(): 
            item[k] = pms_transforms.arrayToTensor(item[k])

        if self.args.in_light:
            item['light'] = torch.from_numpy(lights).view(-1, 1, 1).float()
             
        return item

    def __len__(self):
        return len(self.shape_list)
