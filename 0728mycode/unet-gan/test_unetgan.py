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
from models  import UNet3d_vae
from loss    import loss_3d_crossentropy ,loss_3d_f1_score

train_path = '../../vae/brats18_data/train_2/'
type1 = ['flair','t1','t1ce','t2']

valid_index = np.load('valid.npy')
# valid_index = valid_index[:2] 
print(valid_index)

batch_size = 1
workers = 2

valid_set = tumor_dataset(path = train_path,out_index=valid_index)
valid_loader = DataLoader(valid_set, batch_size=batch_size,shuffle=False, num_workers=workers)
X = 240
Y = 240
Z = 155
x = 104
y = 104
z = 104
model = Generator(4,classes,104,104,104)
print(model)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001,betas=(0.5, 0.999))
# criterion = loss_3d_crossentropy(5,120,120,152)



loss = 0
score = 0


def test_unetgan (model , dataloader ,X,Y,Z, x, y ,z) :
    model.eval()
    macro_f1 = 0
    with torch.no_grad():
        for idx ,(feat,gt) in enumerate (dataloader):
            feat = feat.cuda()
            p = np.zeros((1,)) 
            g = np.zeros((1,))
            for i in range (0,X,x):
                for j in range (0,Y,y):
                    for k in range (0,Z,z):
                        x1 = i
                        cut_x1 = 0
                        x2 = i+x
                        if (x2>X):
                            x2 = X
                            tp_x1 = x1
                            x1 = x2-x
                            cut_x1 = tp_x1-x1
                        y1 = j
                        cut_y1 = 0
                        y2 = j+y
                        if (y2>Y):
                            y2 = Y
                            tp_y1 = y1
                            y1 = y2-y
                            cut_y1 = tp_y1-y1
                        z1 = k
                        cut_z1 = 0
                        z2 = k+z
                        if (z2>Z):
                            z2 = Z
                            tp_z1 = z1
                            z1 = z2-z 
                            cut_z1 = tp_z1-z1 
                        feat_cut = feat[:,:,x1:x2,y1:y2,z1:z2]
                        gt_cut = gt[:,x1:x2,y1:y2,z1:z2]
                        _,pred,__,___ = model(feat_cut)
                        pred = torch.argmax(pred,1).cpu()
                        pred  = pred[:,cut_x1:,cut_y1:,cut_z1:]
                        gt_cut = gt_cut[:,cut_x1:,cut_y1:,cut_z1:]
                        pred   = pred.numpy().reshape(-1)
                        gt_cut = gt_cut.numpy().reshape(-1)
                        p = np.concatenate((p,pred))       
                        g = np.concatenate((g,gt_cut))  
            p = p[1:]                                                                                   
            g = g[1:]
            macro_f1+=f1_score(g, p,average='macro')
            print(idx, p.shape ,end='\r')
        macro_f1 = macro_f1/len(dataloader)
    return macro_f1

#    

cp_path  = 'unetgan_104^3'
txt_path = "result/%s.txt"%cp_path

for epoch in range (41,44) :
    print(epoch , end='\r')
# score , loss = test(model,valid_loader,criterion,120,120,152)
    checkpoint_path = './savecheckpoint_104/G_net_epoch%s.pth'%(epoch)
    load_checkpoint(checkpoint_path,model,optimizer)
    score = test_unetgan(model,valid_loader,X,Y,Z,x,y,z)
    print('[--%s--score:%.4f,loss:%.4f'%(checkpoint_path,score,loss))
    text_file = open(txt_path, "a")
    text_file.write( '%d,%.6f,%.6f\n'%(epoch,score,loss))
    text_file.close()



def test_vae_backup (model , dataloader ,X,Y,Z, x, y ,z) :
    model.eval()
    macro_f1 = 0
    with torch.no_grad():
        for idx ,(feat,gt) in enumerate (dataloader):
            print(idx ,end='\r')
            feat = feat.cuda()
            p = np.zeros((1,)) 
            g = np.zeros((1,))
            for i in range (0,X,64):
                for j in range (0,Y,64):
                    for k in range (0,Z,64):
                        x1 = i
                        x2 = i+x
                        if (x2>=X):
                            x2 = X-1
                            x1 = x2-x
                        y1 = j
                        y2 = j+y
                        if (y2>=Y):
                            y2 = Y-1
                            y1 = y2-y
                        z1 = k
                        z2 = k+z
                        if (z2>=Z):
                            z2 = Z-1
                            z1 = z2-z  
                        feat_cut = feat[:,:,x1:x2,y1:y2,z1:z2]
                        gt_cut = gt[:,x1:x2,y1:y2,z1:z2].numpy().reshape(-1)
                        _,pred,__,___ = model(feat_cut)
                        pred = torch.argmax(pred,1).cpu().numpy().reshape(-1)
                        p = np.concatenate((p,pred))       
                        g = np.concatenate((g,gt_cut))     
            p = p[1:]                                                                                   
            g = g[1:]
            macro_f1+=f1_score(g, p,average='macro')
        macro_f1 = macro_f1/len(dataloader)
    return macro_f1

