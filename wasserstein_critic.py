import torch
from torch import nn
import torchvision

## BUILDING DISCRIMINATOR
class Wasserstein_distance(nn.Module):
    def __init__(self, channels_image, features_d):
        super(Wasserstein_distance, self).__init__()
        
        # Input: N * channels_image * 64 * 64
        self.disc = nn.Sequential(
            
            ## ----------LAYER-1----------
            nn.Conv2d(in_channels=channels_image,
                      out_channels=features_d,
                      kernel_size=4,
                      stride=2,
                      padding=1 
            ), ## 32x32

            nn.LeakyReLU(0.2),
            
            ## ----------LAYER-2----------
            self.nn_block(
                      in_channels=features_d,
                      out_channels=features_d*2,
                      kernel_size=4,
                      stride=2,
                      padding=1
            ), ## 16x16
            
            ## ----------LAYER-3----------
            self.nn_block(
                      in_channels=features_d*2,
                      out_channels=features_d*4,
                      kernel_size=4,
                      stride=2,
                      padding=1
            ), ## 8x8
            
            ## ----------LAYER-4----------
            self.nn_block(
                      in_channels=features_d*4,
                      out_channels=features_d*8,
                      kernel_size=4,
                      stride=2,
                      padding=1
            ), ## 8x8
            
            ## ----------LAYER-5----------
            nn.Conv2d(
                      in_channels=features_d*8,
                      out_channels=1,
                      kernel_size=4,
                      stride=2,
                      padding=0
            ), ## 1x1            
        )
        
    def nn_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=False
                     ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.disc(x)