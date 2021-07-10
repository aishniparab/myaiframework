import torch
from torch import nn

class Linear(torch.nn.Module):
    """
        * input: image vector of size (feat_dim, 3)
        * output: predictions of dim (batch_size*num_samples*num_labels, 2)
        * feat_dim = embedding_output_dim + concat one-hot labels of size 3 = 128+3 = 131
        * if resnet embedding is turned off feat_dim = 3
    """
    def __init__(self, random_seed=123, in_dim=131, out_dim=2): #, device=None):
        torch.manual_seed(random_seed)
        super(Linear, self).__init__()

        self.linear_layer = nn.Linear(in_dim, out_dim) 

    def forward(self, x, edge_index):
        h = None
        out = self.linear_layer(x)
        
        return out, h