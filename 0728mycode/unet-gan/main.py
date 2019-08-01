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

#from models  import unet_3d
from unet_gan  import *
from loss    import loss_3d_crossentropy ,F1_Loss

#set gpu
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


train_path = '../vae/brats18_data/train_2/'
type1 = ['flair','t1','t1ce','t2']
batch_size = 1
workers = 2
classes = 5
x = 48 ; y = 48; z = 48
train_set = tumor_dataset(path = train_path)
train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=False, num_workers=workers)
valid_set = tumor_dataset(path = train_path)
valid_loader = DataLoader(valid_set, batch_size=batch_size,shuffle=False, num_workers=workers)


net_G = Generator(4,classes,48,48,48)
print(net_G)
net_G=net_G.cuda()

net_D = Discriminator()
print(net_D)
net_D=net_D.cuda()

optimizerG = torch.optim.Adam(net_G.parameters(), lr=0.00001, betas=(0.5, 0.999))
optimizerD = torch.optim.Adam(net_D.parameters(), lr=0.00001, betas=(0.5, 0.999))

#criterion = nn.BCELoss()
#criterion  = nn.MSELoss()
#criterion2 = loss_3d_crossentropy(classes,x,y,z)


#load_checkpoint('./savecheckpoint3/_epoch76.pth',model,optimizer)
#print('load epoch49')


train(net_G,net_D,optimizerG,optimizerD, train_loader,'./savecheckpoint/' , x,y,z,n_epochs = 100 , times = 4,start_epoch = 0 )



feat , gt = next(iter(train_loader))
feat_cut , gt_cut = cut_feat_gt(feat,gt,120,120,120)

