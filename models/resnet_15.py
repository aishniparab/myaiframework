# backbone architecture as stated in the paper
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_planes, out_planes, stride=1, padding=0):
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=padding, bias=False)

def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)

class ResBlock(nn.Module):
  def __init__(self, in_channels, filters):
    super(ResBlock, self).__init__()
    self.relu = nn.LeakyReLU()

    self.conv1 = conv3x3(in_channels, filters)
    self.bn1 = nn.BatchNorm2d(filters)
    
    self.conv2 = conv3x3(filters, filters)
    self.bn2 = nn.BatchNorm2d(filters)
    
    self.conv3 = conv3x3(filters, filters)
    self.bn3 = nn.BatchNorm2d(filters)

    self.maxpool = nn.MaxPool2d(2, padding=1)
    #self.dropout = nn.Dropout(p=0.9)

    self.downsample = conv1x1(in_channels, filters)
                                     
    self.bn4 = nn.BatchNorm2d(filters)

    
  def forward(self, x):    
    residual = self.downsample(x) 
    residual = self.bn4(residual)
    
    # layer 1
    out = self.conv1(x) 
    out = self.bn1(out) 
    out = self.relu(out) 
    
    # layer 2
    out = self.conv2(out) 
    out = self.bn2(out)
    out = self.relu(out)
    
    # layer 3
    out = self.conv3(out) 
    out = self.bn3(out)
    
    out += residual     
    out = self.relu(out)
    out = self.maxpool(out)
    #out = self.dropout(out)
    return out

class ResNet15(nn.Module):
  """
  Backbone model as described in paper ResNet-15 with feature map sizes 32, 64, 
  128, 256, 512 and output feature dimension 128
  """
  def __init__(self, in_channels=1): 
    super(ResNet15, self).__init__()
    # 5 feature maps [32, 64, 128, 256, 512]
    self.block1 = ResBlock(in_channels, 32) 
    self.block2 = ResBlock(32, 64)
    self.block3 = ResBlock(64, 128)
    self.block4 = ResBlock(128, 256)
    self.block5 = ResBlock(256, 512) # 17 x 17
    self.conv1 = conv3x3(512, 128, stride=2) 
    #self.conv2 = conv3x3(128, 1) 
    self.maxpool = nn.MaxPool2d(3, padding=1) # 128 x 1 x 1

  def forward(self, x, weights=None):
    if weights is None:
      #print("im here")
      x = self.block1(x)
      #print('block1', x.shape)
      x = self.block2(x)
      #print('block2', x.shape)
      x = self.block3(x)
      #print('block3', x.shape)
      x = self.block4(x)
      #print('block4', x.shape)
      x = self.block5(x)
      #print('block5', x.shape)
      x = self.maxpool(x)
      #print('maxpool1', x.shape)
      x = self.conv1(x)
      #print('conv1', x.shape)
      
      #x = self.conv2(x)
      #print('conv2', x.shape)
      x = self.maxpool(x)
      #print('maxpool2', x.shape)
    else:
      raise ValueError('Not implemented yet')
    return x