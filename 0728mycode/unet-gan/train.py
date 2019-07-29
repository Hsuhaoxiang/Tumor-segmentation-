import numpy as np
import torch
import torch.nn as nn
from loss    import loss_3d_crossentropy ,F1_Loss

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from loss import vae_loss

real_label = 1
fake_label = 0
def train(model,G,D,optimizer,optimizerG ,optimizerD, dataloader, checkpoint_path,x,y,z,n_epochs = 100 , times = 4 ,start_epoch = 0 ):
    print("available gup number:",torch.cuda.device_count())
    
    best_loss = np.inf
    model.train()
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
                feat_cut ,gt_cut = cut_feat_gt(feat,gt,x,y,z)                
                feat_cut = feat_cut.cuda()
                gt_cut   = gt_cut.cuda()
                model.zero_grad()
                featuremap,pred= model(feat_cut)#featuremap is input for G
               
                
                loss = criterion3(pred.double(),gt_cut).float()
                loss.backward()
                optimizer.step()
                n_loss+=loss.item()

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train with real
                D.zero_grad()
                label = [real_label]

                output = netD(feat_cut)
                errD_real = criterion1(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                # train with fake
                
                fake = G(featuremap)
                label = [fake_label]
                output = D(fake.detach())
                errD_fake = criterion1(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                D_loss+=D_G_z1
                errD = errD_real + errD_fake
                optimizerD.step()

                ############################
                # (1) Update G network: maximize log(D(G(z)))
                ###########################
                G.zero_grad()
                label=[real_label]  # fake labels are real for generator cost
                output = netD(fake)
                errG = criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                G_loss+=D_G_z1
                optimizerG.step()





                #if i==1 and idx==1:
                 #   print("1 1save")
                  #  torch.save(rec,'reconstruct.pt')
                   # torch.save(feat_cut,'feat_cut.pt')
                    #torch.save(pred,'pred.pt')
                    #torch.save(gt_cut,'ground_true.pt')

                print("[%d/%d],[%d/%d],[%d/%d],D_loss :%.4f G_loss: %.4f D(x): %.4f D(G(z)): %.4f / %.4f   unet_loss :%.4f"
                %(epoch+1,n_epochs,i+1,len(dataloader),idx+1,times, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2,loss.item()),end = "\r")
                #print("[%d/%d],[%d/%d],[%d/%d],loss :%.4f"%(epoch+1,n_epochs,i+1,len(dataloader),idx+1,times,loss.item()),end = "\r")
        n_loss/=(len(dataloader)*times)
        G_loss/=(len(dataloader)*times)
        D_loss/=(len(dataloader)*times)
        print("[%d/%d],unet_loss : %.4f  D(G(z))"%(epoch+1,n_epochs,n_loss,D_loss,G_loss))
        # if(n_loss <best_loss):
            # best_loss = n_loss
        save_checkpoint('%s_epoch%d.pth'%(checkpoint_path,epoch+1) ,model ,optimizer )
            # print("save best at epoch:" ,epoch+1)
    save_checkpoint('final_%s.pth'%checkpoint_path ,model ,optimizer )

    
def test(model,dataloader,times= 8):
    model.eval()
    macro_f1 = 0
    with torch.no_grad():
        for i ,(feat,gt) in enumerate (dataloader):
            print(i,end='\r')
            feat  = feat.cuda()
            gt    = gt
            feat1 = feat[:,:,:120,:120,0:152]
            feat2 = feat[:,:,:120,120:,0:152]
            feat3 = feat[:,:,120:,120:,0:152]
            feat4 = feat[:,:,120:,:120,0:152]
            gt1   = gt[:,:120,:120,0:152].numpy().reshape(-1)
            gt2   = gt[:,:120,120:,0:152].numpy().reshape(-1)
            gt3   = gt[:,120:,120:,0:152].numpy().reshape(-1)
            gt4   = gt[:,120:,:120,0:152].numpy().reshape(-1)
            pred1 = model(feat1)
            pred2 = model(feat2)
            pred3 = model(feat3)
            pred4 = model(feat4)
            pred1 = torch.argmax(pred1,1).cpu().numpy().reshape(-1)
            pred2 = torch.argmax(pred2,1).cpu().numpy().reshape(-1)
            pred3 = torch.argmax(pred3,1).cpu().numpy().reshape(-1)
            pred4 = torch.argmax(pred4,1).cpu().numpy().reshape(-1)
            pred1 = np.concatenate((pred1,pred2))
            pred1 = np.concatenate((pred1,pred3))
            pred1 = np.concatenate((pred1,pred4))
            gt1 = np.concatenate((gt1,gt2))
            gt1 = np.concatenate((gt1,gt3))
            gt1 = np.concatenate((gt1,gt4))
            macro_f1+=f1_score(gt1, pred1,average='macro')
    return macro_f1/len(dataloader)

def test_rand(model,dataloader,criterion,x,y,z,times= 8):
    model.eval()
    macro_f1 = 0
    f1_loss  = 0
    with torch.no_grad():
        for i ,(feat,gt) in enumerate (dataloader):
            print(i,end='\r')
            feat  = feat.cuda()
            gt    = gt
            for idx in range (times):
                half_x_length = x//2 
                half_y_length = y//2 
                half_z_length = z//2 
                mid_x = np.random.randint(x//2,feat.shape[2]-x//2)
                mid_y = np.random.randint(y//2,feat.shape[3]-y//2)
                mid_z = np.random.randint(z//2,feat.shape[4]-z//2)
                # print(mid_x,mid_y,mid_z)
                feat_cut = feat[:,:,mid_x-half_x_length:mid_x+half_x_length,mid_y-half_y_length:mid_y+half_y_length,mid_z-half_z_length:mid_z+half_z_length]
                gt_cut   = gt[:,mid_x-half_x_length:mid_x+half_x_length,mid_y-half_y_length:mid_y+half_y_length,mid_z-half_z_length:mid_z+half_z_length]
#                     print(feat_cut.shape,gt_cut.shape)
                feat_cut = feat_cut.cuda()
                gt_cut   = gt_cut.cuda()
                # print(feat_cut.shape,gt_cut.shape)
                pred = model(feat_cut)
                loss = criterion(pred.double(),gt_cut)
                f1_loss+=loss.item()
                gt1   = gt_cut.cpu().numpy().reshape(-1)
                pred = torch.argmax(pred,1).cpu().numpy().reshape(-1)
                macro_f1+=f1_score(gt1, pred,average='macro')
        macro_f1 = macro_f1/(len(dataloader)*times)
        f1_loss  = f1_loss/(len(dataloader)*times)
    return macro_f1 , f1_loss



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
