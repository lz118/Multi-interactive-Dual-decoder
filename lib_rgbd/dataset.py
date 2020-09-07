#coding=utf-8

import os
import cv2
import numpy as np
import torch
try:
    from . import transform
except:
    import transform
from torch.utils.data import Dataset

#commmon trainset
mean_rgb = np.array([[[0.43127787, 0.4015223, 0.44389117]]])*255
std_rgb = np.array([[[0.25044188, 0.25923958, 0.25612995]]])*255

mean_d = np.array([[[0.45592305, 0.45592305, 0.45592305]]])*255
std_d = np.array([[[0.2845027, 0.2845027, 0.2845027]]])*255

#DUTD
#mean_rgb = np.array([[[ 0.4061459, 0.38510114,0.4457303]]])*255
#std_rgb = np.array([[[ 0.25237563, 0.2545061,0.24679454]]])*255
#
#mean_d = np.array([[[0.6786454, 0.6786454, 0.6786454]]])*255
#std_d = np.array([[[0.13604848, 0.13604848, 0.13604848]]])*255
def getRandomSample(rgb,t):
    n = np.random.randint(10)
    zero = np.random.randint(2)
    if n==1:
        if zero:
            rgb = torch.from_numpy(np.zeros_like(rgb))
        else:
            rgb = torch.from_numpy(np.random.randn(*rgb.shape))
    elif n==2:
        if zero:
            t = torch.from_numpy(np.zeros_like(t))
        else:
            t = torch.from_numpy(np.random.randn(*t.shape))
    return rgb.float(),t.float()

class Data(Dataset):
    def __init__(self, root,mode='train'):
        self.samples = []
        lines = os.listdir(os.path.join(root,mode+'_images'))
        self.mode = mode
        for line in lines:
            rgbpath = os.path.join(root,mode+'_images', line)
            tpath = os.path.join(root,mode+'_depth', line[:-4]+'.png')
            maskpath = os.path.join(root,mode+'_masks', line[:-4]+'.png')
            self.samples.append([rgbpath,tpath,maskpath])

        if mode == 'train':
            self.transform = transform.Compose( transform.Normalize(mean1=mean_rgb,std1=std_rgb),
                                                transform.Resize(256 ,256),
                                                transform.RandomHorizontalFlip(),transform.ToTensor())

        elif mode == 'test':
            self.transform = transform.Compose( transform.Normalize(mean1=mean_rgb,std1=std_rgb),
                                                transform.Resize(256,256),
                                                transform.ToTensor())
        else:
            raise ValueError

    def __getitem__(self, idx):
        rgbpath,tpath,maskpath = self.samples[idx]
        rgb = cv2.imread(rgbpath).astype(np.float32)
        t = cv2.imread(tpath).astype(np.float32)
        mask = cv2.imread(maskpath).astype(np.float32)
        H, W, C = mask.shape
        rgb,t,mask = self.transform(rgb,t,mask)
#        if  self.mode == 'train':
#            rgb,t =getRandomSample(rgb,t)
        return rgb,t,mask, (H, W), maskpath.split('/')[-1]

    def __len__(self):
        return len(self.samples)



if __name__=='__main__':
    data = Data('E:\VT5000\VT5000_clearall')
    for i,ba in enumerate(data):
        print(ba)