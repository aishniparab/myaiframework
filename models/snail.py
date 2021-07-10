# backbone architecture as stated in the paper
import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet_15 import * 
from snail_blocks import * 

class SnailFewShot(nn.Module):
  def __init__(self, N=2, K=6, use_cuda=True):
    super(SnailFewShot, self).__init__()
    # N-way, K-shot
    self.encoder = ResNet15()
    num_channels = 131 # feat_dim * img_h*img_w + labels
    num_filters = int(math.ceil(math.log(N*K+1, 2)))
    self.attention1 = AttentionBlock(num_channels, 64, 32)
    num_channels += 32
    self.tc1 = TCBlock(num_channels, N * K + 1, 128)
    num_channels += num_filters * 128 #512 #128
    self.attention2 = AttentionBlock(num_channels, 256, 128)
    num_channels += 128
    self.tc2 = TCBlock(num_channels, N*K + 1, 128)
    num_channels += num_filters * 128
    self.attention3 = AttentionBlock(num_channels, 512, 256)
    num_channels += 256
    self.fc = nn.Linear(num_channels, N)
    self.N = N
    self.K = K
    
    #self.use_cuda = use_cuda

  def forward(self, input, labels): #input should be all 8 imgs in the batch not 1 img # assume batch size 1 for now
    #print('input', input.shape, 'labels', labels.shape) # input [112, 1, 32, 32], labels [1, 8, 7, 2, 3]
    #b, k, n, d = input.shape 
    num_samples, channels, img_h, img_w = input.shape
    _, b, k, n, labels_dim = labels.shape
    # ResNet15 Input Shape (sammples x channels x img_dim)
    x = self.encoder(input)
    #print('bongard output shape', x.shape)
    x = x.view(-1, 128)
    labels = labels.view(-1, 3)
    #print("unlabled dim shape", labels.shape)
    
    #x = x.view(b, k, n, ) # 1, 7, 2, 17*17*128 print this
    #print('x before cat', x.shape)
    
    x = torch.cat([x, labels], -1) 
    #print('x after cat', x.shape)
    x = x.view(b, k*n, -1) # should be: 1, 14, 131; actually is: 1, 14, 128
    #print('x before attn', x.shape)
    x = self.attention1(x)
    x = self.tc1(x)
    x = self.attention2(x)
    x = self.tc2(x)
    x = self.attention3(x)
    x = self.fc(x)
    #print(x.isnan().any())
    #if (x.isnan().any()):
    #print(x)
    #print('model output shape: ', x.shape)
    out = x.view(b, k, n, n) # what is the shape here?
    #print('model output reshape: ', out.shape)
    return out