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
from loss    import loss_3d_crossentropy ,loss_3d_f1_score

train_path = 'brats18_data/train_2/'
txt_path = "120120152_f1.txt"
type1 = ['flair','t1','t1ce','t2']

valid_index = np.load('120120152_ce/valid.npy')
# valid_index = valid_index[:2] 
print(valid_index)

batch_size = 1
workers = 2

valid_set = tumor_dataset(path = train_path,out_index=valid_index)
valid_loader = DataLoader(valid_set, batch_size=batch_size,shuffle=False, num_workers=workers)

model = UNet3d(4,5)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001,betas=(0.5, 0.999))
criterion = loss_3d_crossentropy(5,120,120,152)



loss = 0
score = 0

for epoch in [1,2,4,7] :
    print(epoch)
# score , loss = test(model,valid_loader,criterion,120,120,152)
    checkpoint_path = '120120152_f1/epoch%d_120^2_152_f1.pth'%epoch

    load_checkpoint(checkpoint_path,model,optimizer)


    score = test(model,valid_loader)

    print('[--%s--score:%.4f,loss:%.4f'%(checkpoint_path,score,loss))

    text_file = open(txt_path, "a")

    text_file.write( '%d,%.6f,%.6f\n'%(epoch,score,loss))

    text_file.close()

# for epoch in [15,16,17,18,19] :
#     print(epoch)
# # score , loss = test(model,valid_loader,criterion,120,120,152)
#     checkpoint_path = '120120152_ce/120^2_152_ce_epoch%d.pth'%epoch

#     load_checkpoint(checkpoint_path,model,optimizer)


#     score = test(model,valid_loader)

#     print('[--%s--score:%.4f,loss:%.4f'%(checkpoint_path,score,loss))

#     text_file = open(txt_path, "a")

#     text_file.write( '%d,%.6f,%.6f\n'%(epoch,score,loss))

#     text_file.close()