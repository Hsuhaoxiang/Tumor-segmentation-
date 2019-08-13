import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(1), -1)


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x



class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
#         print(x.shape)
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2),
            double_conv(in_ch, out_ch)
        )
    def forward(self, x):
        x = self.mpconv(x)
        return x



class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
#         print(x1.shape)
        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
#         print(x1.shape)
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x



class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x



class Generator (nn.Module):
    def __init__(self, n_channels, n_classes,x,y,z):
        super(Generator, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 256)
        self.up1 = up(512, 128)
        self.up2 = up(256, 64)
        self.up3 = up(128, 64)
        self.outc = outconv(64, n_classes)
        
        #self.flattenlayer = Flatten()
        self.up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                    nn.Conv3d(256, 128, 3, padding=1),
                    nn.BatchNorm3d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(128, 64, 3, padding=1),
                    nn.BatchNorm3d(64),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                    nn.Conv3d(64, 64, 3, padding=1),
                    nn.BatchNorm3d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(64, 32, 3, padding=1),
                    nn.BatchNorm3d(32),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                    nn.Conv3d(32, 16, 3, padding=1),
                    nn.BatchNorm3d(16),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(16, 4, 3, padding=1),
                    nn.BatchNorm3d(4),
                    nn.ReLU(inplace=True),
                    
                    

                    
        )

    def forward(self, x):
        #encoder
        #print("input",x.size())
        x1 = self.inc(x)
        #print("x1 size is",x1.size())
        x2 = self.down1(x1)
        #print("x2 size is",x2.size())
        x3 = self.down2(x2)
        #print("x3 size is",x3.size())
        x4 = self.down3(x3)
        #print("x4 size is",x4.size())
        #decoder segment
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        x = F.softmax(x , 1)
        #reconstruct img
        #x4 = x4.view(-1,8*8*8*256)
        #x4 = self.vae (x4)
        #mu = self._enc_mu(x4)
        #log_sigma = self._enc_log_sigma(x4)
        #sigma = torch.exp(log_sigma)
        #std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().cuda()
        #z =  mu + sigma * std_z # Reparameterization trick
        #print("x4",x4.size())
        recimg=self.up(x4)
        #print("recimg",recimg.size())
        return recimg, x


class Discriminator (nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        #self.model = nn.Sequential(
         #   inconv(4, 64),
          #  down(64, 128),
           # down(128, 256),
           # down(256, 256),
           # nn.Linear(256,1),
           # nn.Sigmoid(),
        #)
       
        self.inc = inconv(4, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 256)
        self.linearlayer1 = nn.Linear(256,1)
        self.linearlayer2 = nn.Linear(216,1)
        self.linearlayer3 = nn.Linear(8,1)
        self.end = nn.Sigmoid()
        
    



    def forward(self, img):
        img = self.inc(img)
        img = self.down1(img)
        img = self.down2(img)
        img = self.down3(img)
        img = img.view(-1,256)
        #print(img.size())
        img = self.linearlayer1(img)
        img = img.view(-1,216)
        #print(img.size())
        img = self.linearlayer2(img)
        #print(img.size())
        #img = img.view(-1,8)
        #img = self.linearlayer3(img)
        validity = self.end(img)
   
        
        return validity


