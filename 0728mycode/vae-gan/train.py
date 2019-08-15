import numpy as np
import torch
import torch.nn as nn
from loss    import loss_3d_crossentropy ,F1_Loss

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from loss import vae_loss
import os
import csv
real_label = 1
fake_label = 0
def train(G,D,optimizerG ,optimizerD, dataloader, checkpoint_path,x,y,z,n_epochs = 100 , times = 4 ,start_epoch = 0 ):
    
    print("available gup number:",torch.cuda.device_count())

    best_loss = np.inf
    G.train()
    D.train()
    criterion1 = nn.BCELoss()
    criterion2  = nn.MSELoss()
    criterion3=loss_3d_crossentropy(5,x,y,z)
    for epoch in range (start_epoch , n_epochs):
        n_loss = 0
        G_loss = 0
        D_loss = 0
        for i ,(feat,gt) in enumerate (dataloader):        
            # print(feat.shape,gt.shape)
            for idx in range (times):


                #data process
                feat_cut ,gt_cut = cut_feat_gt(feat,gt,x,y,z)                
                feat_cut = feat_cut.cuda()
                gt_cut   = gt_cut.cuda()


                """
                featuremap,pred= model(feat_cut)#featuremap is input for G            
                loss = criterion3(pred.double(),gt_cut).float()
                loss.backward()
                optimizer.step()
                n_loss+=loss.item()
                """
                
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train with real
                D.zero_grad()
                G.zero_grad()
                label = torch.full((1,1), real_label)
                label=label.cuda()
                
                
                
                
                output = D(feat_cut)
                #print(output.size())
                #print(label.size())
                errD_real = criterion1(output, label)
                #print("errD: ",errD_real)
                errD_real.backward()
                D_x = output.mean().item()

                # train with fake
                
                fake ,pred= G(feat_cut)
                label = torch.full((1,), fake_label)
                label=label.cuda()
                
                output = D(fake.detach())
                output = output.squeeze()
                errD_fake = criterion1(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                
                errD = errD_real + errD_fake
                D_loss+=errD
                optimizerD.step()

                ############################
                # (1) Update G network: maximize log(D(G(z)))
                ###########################
                
                label = torch.full((1,), real_label)  # fake labels are real for generator cost
                label=label.cuda()                        
                output = D(fake)    
                loss = criterion3(pred.double(),gt_cut).float()
                n_loss+=loss.item()  
                errG = criterion1(output, label)
                Gnet_loss=errG+loss
                Gnet_loss.backward()
                G_loss+=errG
                D_G_z2 = output.mean().item()                
                optimizerG.step()
                ###################################
                #    Finish update G,D network    #
                ###################################
                """
                if i==1 and idx==1:
                    print("1 1save")
                    torch.save(rec,'reconstruct.pt')
                    torch.save(feat_cut,'feat_cut.pt')
                    torch.save(pred,'pred.pt')
                    torch.save(gt_cut,'ground_true.pt')
                """
                print("[%d/%d],[%d/%d],[%d/%d],D_loss :%.4f G_loss: %.4f D(x): %.4f D(G(z)): %.4f / %.4f   unet_loss :%.4f"
                %(epoch+1,n_epochs,i+1,len(dataloader),idx+1,times, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2,loss.item()),end = "\r")
                #print("[%d/%d],[%d/%d],[%d/%d],loss :%.4f"%(epoch+1,n_epochs,i+1,len(dataloader),idx+1,times,loss.item()),end = "\r")
        n_loss/=(len(dataloader)*times)
        G_loss/=(len(dataloader)*times)
        D_loss/=(len(dataloader)*times)
        print("[%d/%d],unet_loss : %.4f  D_loss :%.4f G_loss: %.4f"%(epoch+1,n_epochs,n_loss,D_loss,G_loss))
        with open('output.csv','a',newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch+1,n_epochs,n_loss,D_loss,G_loss])
        csvfile.close()
        # if(n_loss <best_loss):
            # best_loss = n_loss
        save_checkpoint('%sG_net_epoch%d.pth'%(checkpoint_path,epoch+1) ,G ,optimizerG )
        save_checkpoint('%sD_net_epoch%d.pth'%(checkpoint_path,epoch+1) ,D ,optimizerD )
            # print("save best at epoch:" ,epoch+1)
    save_checkpoint('%sG_netfinal_.pth'%checkpoint_path ,G ,optimizerG)
    save_checkpoint('%sD_netfinal_.pth'%checkpoint_path ,D ,optimizerD )



def save_checkpoint(checkpoint_path,model,optimizer):
    state = {'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)

def load_checkpoint(checkpoint_path,model,optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    #print('model loaded from %s' % checkpoint_path)

def cut_train_val (total_len, train_len):
    ##cut train and val
    arr = np.arange(total_len)
    np.random.shuffle(arr)
    train_index = arr[:train_len]
    valid_index   = arr[train_len:]
    np.save('train.npy', train_index)
    np.save('valid.npy', valid_index)
    return train_index , valid_index

def normalize (ndarray):
    std = np.std(ndarray)
    mean = np.mean(ndarray)
    ndarray = (ndarray-mean)/std
    return ndarray


def cut_feat_gt (feat, gt,x,y,z):
    half_x_length = x//2 
    half_y_length = y//2 
    half_z_length = z//2 
    mid_x = np.random.randint(x//2,feat.shape[2]-x//2)
    mid_y = np.random.randint(y//2,feat.shape[3]-y//2)
    mid_z = np.random.randint(z//2,feat.shape[4]-z//2)
    # print(mid_x,mid_y,mid_z)
    feat_cut = feat[:,:,mid_x-half_x_length:mid_x+half_x_length,mid_y-half_y_length:mid_y+half_y_length,mid_z-half_z_length:mid_z+half_z_length]
    gt_cut   = gt[:,mid_x-half_x_length:mid_x+half_x_length,mid_y-half_y_length:mid_y+half_y_length,mid_z-half_z_length:mid_z+half_z_length]
    return feat_cut , gt_cut
