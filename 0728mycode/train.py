import numpy as np
import torch
import torch.nn as nn
from loss    import loss_3d_crossentropy ,F1_Loss

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from loss import vae_loss


def train(model,optimizer, dataloader, checkpoint_path,x,y,z,n_epochs = 100 , times = 4 ,start_epoch = 0 ):
    print("available gup number:",torch.cuda.device_count())
    
    best_loss = np.inf
    model.train()
    criterion  = nn.MSELoss()
    criterion2=loss_3d_crossentropy(5,x,y,z)
    for epoch in range (start_epoch , n_epochs):
        n_loss = 0
        for i ,(feat,gt) in enumerate (dataloader):        
            # print(feat.shape,gt.shape)
            for idx in range (times):
                feat_cut ,gt_cut = cut_feat_gt(feat,gt,x,y,z)                
                feat_cut = feat_cut.cuda()
                gt_cut   = gt_cut.cuda()
                model.zero_grad()
                rec,pred ,mu,var= model(feat_cut)
                rec_i=rec.contiguous().view(-1)
                feat_i= feat_cut.contiguous().view(-1) 
                
                loss =loss = 0.1*criterion(rec_i , feat_i)+criterion2(pred.double(),gt_cut).float()
                #loss = vae_loss(pred,feat_cut , mu, var,x,y,z)
                #loss = criterion(pred_i.double(),feat_i.double())
                loss.backward()
                optimizer.step()
                n_loss+=loss.item()

                #if i==1 and idx==1:
                 #   print("1 1save")
                  #  torch.save(rec,'reconstruct.pt')
                   # torch.save(feat_cut,'feat_cut.pt')
                    #torch.save(pred,'pred.pt')
                    #torch.save(gt_cut,'ground_true.pt')


                print("[%d/%d],[%d/%d],[%d/%d],loss :%.4f"%(epoch+1,n_epochs,i+1,len(dataloader),idx+1,times,loss.item()),end = "\r")
        n_loss/=(len(dataloader)*times)
        
        print("[%d/%d],loss : %.4f"%(epoch+1,n_epochs,n_loss))
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
