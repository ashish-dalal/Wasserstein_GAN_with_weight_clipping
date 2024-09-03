import torch
from torch import nn
import torchvision

class Generator(nn.Module):
    def __init__(self, z_dim, channels_image, features_g):
        super(Generator, self).__init__()
        
        ## INPUT: N x z_dim x 1 x 1
        self.net = nn.Sequential(
            
            ## ----------LAYER-1----------
            self.nn_block(
                in_channels=z_dim,
                out_channels=features_g*16,
                kernel_size=4,
                stride=1,
                padding=0
            ), # N x features_g*16 x 4 x 4
            
            ## ----------LAYER-2----------
            self.nn_block(
                in_channels=features_g*16,
                out_channels=features_g*8,
                kernel_size=4,
                stride=2,
                padding=1
            ), # N x features_g*16 x 8 x 8
            
            ## ----------LAYER-3----------
            self.nn_block(
                in_channels=features_g*8,
                out_channels=features_g*4,
                kernel_size=4,
                stride=2,
                padding=1
            ), # N x features_g*16 x 16 x 16
            
            ## ----------LAYER-4----------
            self.nn_block(
                in_channels=features_g*4,
                out_channels=features_g*2,
                kernel_size=4,
                stride=2,
                padding=1
            ), # N x features_g*16 x 32 x 32
            
            ## ----------LAYER-5----------
            nn.ConvTranspose2d(
                in_channels=features_g*2,
                out_channels=channels_image,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            
            nn.Tanh()
        )
        
    def nn_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)