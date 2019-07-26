# import library
import nibabel as nib
import numpy as np
import os
import torch
import datetime
#pytorch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

#torcvision
import torchvision.models as models
import torchvision.transforms as transforms

#import python file
from dataset import tumor_dataset
from train   import *
from models  import UNet3d
from models  import UNet3d_vae
from loss    import loss_3d_crossentropy ,F1_Loss


train_path = 'brats18_data/train_2/'
type1 = ['flair','t1','t1ce','t2']



batch_size = 1
workers = 2
classes = 5
x = 120 ; y = 120; z = 152

train_set = tumor_dataset(path = train_path)
train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=False, num_workers=workers)
valid_set = tumor_dataset(path = train_path)
valid_loader = DataLoader(valid_set, batch_size=batch_size,shuffle=False, num_workers=workers)

model = UNet3d_vae(4,classes)
print(model)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001,betas=(0.5, 0.999))
#criterion = loss_3d_crossentropy(classes,x,y,z)
criterion = nn.MSELoss(size_average = False)

load_checkpoint('./_epoch48.pth',model,optimizer)
print("load_checkpoint")
train(model,optimizer, train_loader ,criterion,'./savecheckpoint/' , x,y,z,n_epochs = 100 , times = 4,start_epoch = 48 )



feat , gt = next(iter(train_loader))
feat_cut , gt_cut = cut_feat_gt(feat,gt,120,120,120)

