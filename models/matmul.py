import torch
from torch import nn
import math

# this doesn't work
class MatMul(torch.nn.Module):
    """
        * input: image vector of size (feat_dim, 3)
        * output: predictions of dim (batch_size*num_samples*num_labels, 2)
        * feat_dim = embedding_output_dim + concat one-hot labels of size 3 = 128+3 = 131
        * if resnet embedding is turned off feat_dim = 3
    """
    def __init__(self, random_seed=123, in_dim=131, out_dim=2): #, device=None):
        torch.manual_seed(random_seed)
        super(MatMul, self).__init__()
        #self.projection_vector = torch.tensor([3, out_dim], dtype=torch.float32)
        #self.projection_vector = nn.Mul()
        self.scalar = math.pi

    def forward(self, x, edge_index):
        h = None
        #out = torch.matmul(x, self.projection_vector) # input * vector = scalar
        out = torch.mul(x, self.scalar)
        print(out)
        return out, h