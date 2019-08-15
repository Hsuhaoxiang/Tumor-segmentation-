import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

class loss_3d_crossentropy(nn.Module):
    def __init__(self,class_num,x,y,z):
        super(loss_3d_crossentropy,self).__init__()
        self.class_num = class_num
        self.x = x
        self.y = y
        self.z = z
    def forward(self,pred_tensor,target_tensor):
        voxel = self.x*self.y*self.z
        pred1 = pred_tensor.view(-1,self.class_num,voxel)
        gt1   = target_tensor.view(-1,voxel)
        gt1   = gt1.long()
        loss = F.cross_entropy(pred1,gt1)
        return loss
    
class loss_3d_f1_score(nn.Module):
    def __init__(self):
        super(loss_3d_f1_score,self).__init__()
        
    def forward(self,pred_tensor,target_tensor):
        pred = torch.argmax(pred_tensor,1).cpu().detach().numpy()
        target = target_tensor.cpu().detach().numpy()
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        f1 = f1_score(target, pred,average='macro')
        loss = torch.tensor(1-f1,requires_grad=True)
        return loss

def vae_loss (recon_img,img , mu, logvar,x,y,z):
    recon_img  = recon_img.view(-1,4*x*y*z)
    img   = img.view(-1,4*x*y*z)
    BCE = F.mse_loss(recon_img,img , size_average=False)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


class F1_Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(F1_Loss, self).__init__()
        self.epsilon = epsilon
 
    def forward(self, output, target):
        probas = nn.Sigmoid()(output)
        TP = (probas * target).sum(dim=1)
        precision = TP / (probas.sum(dim=1) + self.epsilon)
        recall = TP / (target.sum(dim=1) + self.epsilon)
        f1 = 2 * precision * recall / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()
