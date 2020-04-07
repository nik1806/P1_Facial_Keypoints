## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

# auto padding to preserve the shape
class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

# stack of conv and batchnorm for repeated use
def conv_bn(in_channels, out_channels, *args, **kwargs):
    return nn.Sequential(Conv2dAuto(in_channels, out_channels, *args, **kwargs),
                         nn.BatchNorm2d(out_channels))

class ResnetResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, downsampling=1, activation=nn.SELU()):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.activate = activation
        self.blocks = nn.Sequential(conv_bn(in_channels, out_channels, kernel_size=kernel, stride=downsampling, bias=False),
                       self.activate,
                       conv_bn(out_channels, out_channels, kernel_size=kernel, bias=False))
        self.shortcut = conv_bn(in_channels, out_channels, kernel_size=1, stride=downsampling, bias=False) if self.should_apply_shortcut else None
        
        
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels
    
class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        # (in dim) 1x224x224 -> (op dim) 32x218x218
        self.conv1 = nn.Conv2d(1, 32, 7)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2,2) 
        # (in dim) 32x109x109 -> (out dim) 64x55x55
        self.res1 = ResnetResidualBlock(32, 64, 5, 2)
        # (in dim) 64x55x55 -> (out dim) 128x28x28
        self.res2 = ResnetResidualBlock(64, 128, 3, 2)
        # (in dim) 128x28x28 -> (out dim) 256x14x14
        self.res3 = ResnetResidualBlock(128, 256, 3, 2)
        # (in img) 256x14x14
        self.dense1 = nn.Linear(256*14*14, 2048)
        self.drop1 = nn.Dropout(p=0.4)
        self.dense2 = nn.Linear(2048, 1024)
        self.drop2 = nn.Dropout(p=0.5)
        self.dense3 = nn.Linear(1024, 136)
        
    def forward(self,x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = x.view(x.size(0), -1) # flattening
        x = self.drop1(F.relu(self.dense1(x)))
        x = self.drop2(F.relu(self.dense2(x)))
        x = self.dense3(x)
        
        return x
'''
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # initial image shape = 1x224x224
        self.conv1 = nn.Conv2d(1, 32, 5) # op -> 32x220x220
        self.bn1 = nn.BatchNorm2d(32)
        # input image = 32x110x110 (after match norm) -> 64x107x107(output)
        self.conv2 = nn.Conv2d(32, 64, 4)
        self.bn2 = nn.BatchNorm2d(64)
        # (in img dim) 64x53x53 -> (out img dim) 128x51x51
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)
        # (in img dim) 128x25x25 -> (out img dim) 256x24x24
        self.conv4 = nn.Conv2d(128, 256, 2)
        self.bn4 = nn.BatchNorm2d(256)   
        # (in img) 256x24x24 -> (out img) 256x12x12
        self.dense1 = nn.Linear(256*12*12, 1024)
        self.drop1 = nn.Dropout(p=0.4)
        self.dense2 = nn.Linear(1024, 512)
        self.drop2 = nn.Dropout(p=0.5)
        self.dense3 = nn.Linear(512, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool = nn.MaxPool2d(2,2)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1) # flattening
        x = self.drop1(F.relu(self.dense1(x)))
        x = self.drop2(F.relu(self.dense2(x)))
        x = self.dense3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
'''
