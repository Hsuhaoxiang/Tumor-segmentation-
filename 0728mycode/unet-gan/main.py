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

#from models  import UNet3d
from unet_gan  import *
from loss    import loss_3d_crossentropy ,F1_Loss

#set gpu
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


train_path = '../vae/brats18_data/train_2/'
type1 = ['flair','t1','t1ce','t2']
batch_size = 1
workers = 2
classes = 5
x = 32 ; y = 32; z = 32

train_set = tumor_dataset(path = train_path)
train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=False, num_workers=workers)
valid_set = tumor_dataset(path = train_path)
valid_loader = DataLoader(valid_set, batch_size=batch_size,shuffle=False, num_workers=workers)


unet = unet_3d(4,classes,32,32,32)
print(unet)
unet.cuda()
generator = Genernator()
print(generator)
generator.cuda()
discriminator = Discriminator()
print(discriminator)
discriminator.cuda()

optimizer = torch.optim.Adam(unet.parameters(),lr=0.00001,betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(generator.parameters(), lr=0.00001, betas=(0.5, 0.999))
optimizerD = torch.optim.Adam(discriminator.parameters(), lr=0.00001, betas=(0.5, 0.999))

#criterion = nn.BCELoss()
#criterion  = nn.MSELoss()
#criterion2 = loss_3d_crossentropy(classes,x,y,z)


#load_checkpoint('./savecheckpoint3/_epoch76.pth',model,optimizer)
#print('load epoch49')


train(unet,generator,discriminator,optimizer,optimizerG,optimizerD, train_loader,'./savecheckpoint3/' , x,y,z,n_epochs = 100 , times = 4,start_epoch = 0 )



feat , gt = next(iter(train_loader))
feat_cut , gt_cut = cut_feat_gt(feat,gt,120,120,120)

