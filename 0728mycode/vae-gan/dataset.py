import os
import nibabel as nib
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from train   import normalize

type1 = ['flair','t1','t1ce','t2']

class tumor_dataset(Dataset):
    def __init__(self,path,transform = None,read_label=True,read_image=True):
        self.path = path
        self.list = sorted(os.listdir(self.path))
        
        self.len  = len(self.list)
#         self.transform = transform
        self.read_label = read_label
        self.read_image = read_image
        print('datalen: ',self.len)
    def __getitem__(self, index):
        ##set path
        abs_path = self.path+self.list[index]+'/'+self.list[index]+'_'
        ##read ground truth
        if(self.read_label==True):
            gt_path  = abs_path+'seg.nii.gz'
            gt = nib.load(gt_path)
            gt = gt.get_fdata()
        ##read image and normalize 
        if(self.read_image==True):
            feat     = nib.load(abs_path+type1[0]+'.nii.gz')
            feat     = feat.get_fdata()
            feat     = np.expand_dims(feat, axis=0)
            feat     = normalize(feat)
            for i in range (1,4):
                feat1    = nib.load(abs_path+type1[i]+'.nii.gz')
                feat1    = feat1.get_fdata()
                feat1    = normalize(feat1)
                feat1    = np.expand_dims(feat1, axis=0)
                feat     = np.concatenate((feat,feat1),axis=0)
            feat = torch.tensor(feat).type('torch.FloatTensor')
        if(self.read_image==False):
            return gt
        elif(self.read_label==False):
            return feat
        else:
            return feat ,gt
    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len
