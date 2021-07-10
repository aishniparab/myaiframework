import torch
from torch import nn
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    """
        * input: image vector of size (in_dim, 3)
        * output: predictions of dim (batch_size*num_samples*num_labels, 2)
        * in_dim = embedding_output_dim + concat one-hot labels of size 3 = 128+3 = 131
    """
    def __init__(self, seed=123, in_dim=131, out_dim=2, num_iterations=5): #, device=None):

        torch.manual_seed(seed)

        super(GCN, self).__init__()
         
        self.num_iterations = num_iterations
        
        self.mlp =  nn.Sequential(
                    #nn.Flatten(),
                    nn.Linear(in_dim, 64),
                    nn.Tanh(),
                    nn.Linear(64, 32),
                    nn.Tanh(),
                    nn.Linear(32, 16),
        )
        self.gcn_conv_layers = nn.ModuleList()
        for i in range(self.num_iterations):
            self.gcn_conv_layers.append(GCNConv(16, 16))

        self.conv1 = GCNConv(16, 16) 
        
        self.classifier = nn.Linear(16, out_dim)

    def forward(self, x, edge_index):
        h = self.mlp(x)
        for i in range(self.num_iterations): # message pass for num_iterations
          h = self.conv1(h, edge_index)
          h = h.tanh()  # Final GNN embedding space''
        for gcn_conv in self.gcn_conv_layers:
          h = gcn_conv(h, edge_index)
          h = h.tanh()
        out = self.classifier(h) # Apply a final (linear) classifier.
        
        return out, h

