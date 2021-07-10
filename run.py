import torch
import torch_geometric.utils as utils
import networkx as nx
import numpy as np
from scipy.linalg import null_space
from scipy.sparse import csgraph, linalg
from utils import *
from gcn import *
from resnet_15 import *
from batch_sampler import BatchSampler
from bongard_dataset import BongardDataset
from livelossplot import PlotLosses
import os
from tqdm import tqdm
import time

dataset_dir = '../ShapeBongard_V2'

# use gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GLOBAL GRAPH VARIABLES
N = 2 # num_classes
K = 6 # num_samples_per_class
batch_size = 10 # num bongard problems per epoch (cannot fit more than this in mem)
img_h = 128
img_w = 128
img_dim = (img_h, img_w) 
drop_last=True # assume not same size

# for a single graph G
num_nodes_per_graph = (K + 1) * 2
num_edges_per_graph = num_nodes_per_graph * (num_nodes_per_graph - 1)
send_idx_per_graph = np.arange(num_nodes_per_graph)
receive_idx_per_graph = np.arange(num_nodes_per_graph)
edge_index_per_graph = torch.tensor([[x,y] for x in send_idx_per_graph 
                          for y in receive_idx_per_graph 
                          if x != y]).t().contiguous() # 14 * 13 pairs
G = nx.Graph()
G.add_edges_from(edge_index_per_graph.t().numpy())


# for batch of graphs G * batch_size
num_graphs = batch_size
num_nodes = num_graphs * num_nodes_per_graph
num_edges =  num_graphs * num_edges_per_graph
send_idx = np.split(np.arange(num_nodes), num_graphs)
receive_idx = np.split(np.arange(num_nodes), num_graphs)
edge_index = torch.tensor([[x,y] for i in range(num_graphs)
                                for x in send_idx[i] 
                                for y in receive_idx[i]
                          if x != y]).t().contiguous()       #torch.block_diag(edge_index_per_graph.t(), edge_index_per_graph.t(), dim=)     # stack edge_index_per_graph
G_prime = nx.Graph()
G_prime.add_edges_from(edge_index.t().numpy())

print("Dataset: Bongard-Logo")
print('=======================================================================================')
print("num_classes: ", N)
print("num_samples_per_class: ", K)
print("batch_size: ", batch_size)
print("img_width, img_height: ", img_dim)
print("\n")
"""
print("Single Graph")
print('=========================')
print("num_nodes_per_graph: ", num_nodes_per_graph)
print("num_edges_per_graph: ", num_edges_per_graph)
print("edge_index_per_graph_shape: ", edge_index_per_graph.shape)
#print("nx_laplacian_matrix(): ", nx.laplacian_matrix(G, nodelist=np.arange(num_nodes_per_graph)))
#print("num_of_0_eigen_values: ", linalg.eigs(nx.laplacian_matrix(G, nodelist=np.arange(num_nodes_per_graph)).A, sigma=0, return_eigenvectors=False))
print("num_connected_components(): ", null_space(nx.laplacian_matrix(G, nodelist=np.arange(num_nodes_per_graph)).A).shape[1])
print("num_connected_componenets == batch_size: ", null_space(nx.laplacian_matrix(G, nodelist=np.arange(num_nodes_per_graph)).A).shape[1] == 1)

print("\n")
print("All Graphs")
print('=========================')
print("num_nodes: ", num_nodes)
print("num_edges: ", num_edges)
print("edge_index_shape: ", edge_index.shape)
#print("nx_laplacian_matrix(): ", nx.laplacian_matrix(G_prime, nodelist=np.arange(num_nodes)))
#print("num_of_0_eigen_values: ", linalg.eigs(nx.laplacian_matrix(G_prime, nodelist=np.arange(num_nodes)).A, sigma=0, return_eigenvectors=False))
print("num_connected_components(): ", null_space(nx.laplacian_matrix(G_prime, nodelist=np.arange(num_nodes)).A).shape[1])
print("num_connected_componenets == batch_size: ", null_space(nx.laplacian_matrix(G_prime, nodelist=np.arange(num_nodes)).A).shape[1] == batch_size)
nx.draw(G_prime, node_size=batch_size*0.5, width=0.1, pos=nx.spring_layout(G_prime, k=1/batch_size*4, scale=0.01, seed=1))
plt.savefig('../vis/bongard_graph_setup_networkx.png', dpi=500, bbox_inches = 'tight')
"""

# LOAD DATASET AND BATCH SAMPLER
tr_dataset = BongardDataset(batch_type='train', img_dim=img_dim, batch_size=batch_size, one_hot_size=3, root=dataset_dir)
val_dataset = BongardDataset(batch_type='val', img_dim=img_dim, batch_size=batch_size, one_hot_size=3, root=dataset_dir)
test_dataset = BongardDataset(batch_type='test', img_dim=img_dim, batch_size=batch_size, one_hot_size=3, root=dataset_dir)

tr_sampler = BatchSampler(labels=tr_dataset.y, batch_size=batch_size)
val_sampler = BatchSampler(labels=val_dataset.y, batch_size=batch_size)
test_sampler = BatchSampler(labels=test_dataset.y, batch_size=batch_size)

tr_dataloader = torch.utils.data.DataLoader(tr_dataset, sampler=tr_sampler, drop_last=drop_last)
val_dataloader = torch.utils.data.DataLoader(val_dataset, sampler=val_sampler, drop_last=drop_last)
test_dataloader = torch.utils.data.DataLoader(test_dataset, sampler=test_sampler, drop_last=drop_last)

# MODEL
# to turn off resnet i changed featdim from 131 to 3
model = GCN(batch_dim=batch_size*K+1*N, feat_dim=3, out_dim=1, num_iterations=5) #device=device)
print("\n", model, "\n")

# unit test
learning_rate = 0.001
weight_decay = 5e-4
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
print("measure loss on probe") 
# run couple of times the values are random
for i in range(7):
	print("num flips = ", i)
	batch = next(iter(tr_dataloader))
	data, paths = embed_resnet(batch, i, "probe", edge_index, ResNet15().to(device), device, 0, img_h, img_w)
	loss, acc, out, h = train(data, model, loss_fn, optimizer)
print("===========================================")
print("measure loss on both context and probe")
for i in range(7):
	print("num flips = ", i)
	batch = next(iter(tr_dataloader))
	data, paths = embed_resnet(batch, i, "all", edge_index, ResNet15().to(device), device, 0, img_h, img_w)
	loss, acc, out, h = train(data, model, loss_fn, optimizer)

'''
# init paths to save model, loss and accuracy values
dir_name = get_dirname('gnn_opt_dec_rule_flip_3', 'v9')
export_dir = os.path.join("../trained_models", dir_name)
if not os.path.exists(export_dir): 
  os.makedirs(export_dir)
best_model_path = os.path.join(export_dir, 'best_model.pth')
last_model_path = os.path.join(export_dir, 'last_model.pth')

# init lists to store model state, loss and accuracy values
train_loss = []
train_acc = []
val_loss = []
val_acc = []
best_acc = 0
best_state = None
liveloss = PlotLosses()
embeddings = {}

# hyperparameters
learning_rate = 0.001
#momentum = 0.9
weight_decay = 5e-4
num_epochs = 2
num_tr_iterations = tr_dataset.__len__()//batch_size
num_val_iterations = val_dataset.__len__()//batch_size
num_iterations = 0
encoder = ResNet15().to(device)
model = GCN(batch_size*K+1*N, feat_dim=131, out_dim=1, num_iterations=num_iterations).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) 

print("save_dir: ", export_dir)
print("device: ", device)
print("\n")
print('Hyperparameters')
print('===========================================================================================================')
print("learning_rate: ", learning_rate)
print("weight_decay: ", weight_decay)
print("num_epochs: ", num_epochs)
print("num_tr_iterations: ", num_tr_iterations)
print("num_val_iterations: ", num_tr_iterations)
print("num_message_passing_iterations: ", num_iterations)
print("loss_fn: ", loss_fn.__class__.__name__)
print("optimizer: ", optimizer.__class__.__name__)
print("\n")


for epoch in range(num_epochs):
  print('=== Epoch: {} ==='.format(epoch))
  logs = {}
  for phase in ['train', 'val']:
    if phase == 'train':
      iter_obj = iter(tr_dataloader)
      model.train()
      model = model.to(device)
    else:
      iter_obj = iter(val_dataloader)
      model.eval()
    
    for batch in tqdm(iter_obj):
      data, paths = embed_resnet(batch, 3, edge_index, encoder, device, 0, img_h, img_w)
      if phase == 'train':
        loss, acc, out, h = train(data, model, loss_fn, optimizer)
        train_loss.append(loss.item())
        train_acc.append(acc)
        #if acc == 1.0:
        #	for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'embeddings']:
        #		save_list_to_file(os.path.join(export_dir, name + '.txt'), locals()[name])
        #	break
      else:
        loss, acc, out, h = val(data, model, loss_fn)
        val_loss.append(loss.item())
        val_acc.append(acc)
        #if acc == 1.0:
        #	for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'embeddings']:
        #		save_list_to_file(os.path.join(export_dir, name + '.txt'), locals()[name])
        #	break
    # end for loop over batches
    # compute avg loss and acc over epoch
    if phase == 'train':
      print('Avg Train Loss: {}, Avg Train Acc: {}'.format(np.mean(train_loss[-num_tr_iterations:]), np.mean(train_acc[-num_tr_iterations:]))) 
      logs['loss'] = np.mean(train_loss[-num_tr_iterations:])
      logs['acc'] = np.mean(train_acc[-num_tr_iterations:])
      liveloss.update(logs)
      liveloss.send()
      
    else:
      print('Avg Val Loss: {}, Avg Val Acc: {}'.format(np.mean(val_loss[-num_val_iterations:]), np.mean(val_acc[-num_val_iterations:]))) 
      if epoch % 2 == 0:
          torch.save(model.state_dict(), last_model_path)
      #    embeddings[epoch] = {'h': h, 'color': data.y, 'val_loss': np.mean(val_loss)}
      #    visualize embeddings from last batch
      #    visualize_tsne(h, color=data.y, epoch=epoch, loss=loss, figsize=(7, 7))
      #    time.sleep(0.3)
      if np.mean(val_acc) >= best_acc:
        best_state = model.state_dict()
        torch.save(best_state, best_model_path)
        best_acc = np.mean(val_acc)
      logs['val_loss'] = np.mean(val_loss[-num_val_iterations:])
      logs['val_acc'] = np.mean(val_acc[-num_val_iterations:])
      
      liveloss.update(logs)
      liveloss.send()

    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'embeddings']:
      save_list_to_file(os.path.join(export_dir, name + '.txt'), locals()[name])
    
    torch.cuda.empty_cache() 
    torch.autograd.set_detect_anomaly(True)  
 '''
