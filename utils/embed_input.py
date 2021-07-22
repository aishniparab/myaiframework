import torch
from torch_geometric.data import Data
from utils.train import get_loss_mask
from models.resnet_15 import ResNet15

def embed(batch, num_flip, mask_type, edge_index, device, img_h, img_w, debug_step, embedding_weight, gaussian_sample, vector_dim):
    # refactor this so it does exactly one thing
    '''
    args:
    - batch.x.shape: (batch_num, batch_size, k+1, n, img_h*img_w) 
    - batch.y.shape: (batch_num, batch_size, k+1, n, one_hot_dim=3)
    - num_flip: number of labels to flip
    - mask_probe: bool indicates whether probe should be masked for loss computation
    - debug_step: model input are "gaussian", "resnet_gaussian_labels", "resnet_labels", "labels" 
    returns Data object for model input with flipped 
    - x: flipped labels
    - y: ground truth labels
    - edge_index
    - loss_mask: hides probe labels in x if mask_probe is true
    '''
    
    x, y, paths = batch
    edge_index = edge_index.to(device)
    
    # vary n to verify optimal decision rule 
    loss_mask, y_train = get_loss_mask(y, num_flip, mask_type) # vary n to verify optimal decision rule should not be more than 70% when n=3
    loss_mask = loss_mask.to(device)
    y_train = y_train.to(device) # y_train is one hot with unknown label "0 0 1": (batch_num*batch_size*K+1*N, 3)

    # ground truth labels y
    y = y.view(-1, 3) #y: (batch_num*batch_size*K+1*N, one_hot_dim=3)
    y = y.to(device)
    y = torch.argmax(y, dim=1) # left class = 0, right class = 1
    
    # default: "labels_only" pass y as input to model
    
    if debug_step == "gaussian":
        input_val = torch.cat([gaussian_sample, y_train], -1) 
    elif debug_step == "resnet_gaussian_labels":
        encoder = ResNet15().to(device)    
        x = x.to(device)
        x = x.view(-1, 1, img_h, img_w)
        x = encoder(x)
        x[x<0] = 0
        x = x * embedding_weight
        x = x.squeeze()
        input_val = torch.cat([x, gaussian_sample, y_train], -1) 
    elif debug_step == "resnet_labels":
        encoder = ResNet15().to(device)
        x = x.to(device)
        x = x.view(-1, 1, img_h, img_w) #x: (batch_num*batch_size*K+1*N, channels=1, img_h, img_w)
        x = encoder(x)
        x[x < 0] = 0 #hacky solution -- resnet should not output negative values bcos of relu
        x = x * embedding_weight # knob 
        x = x.squeeze() #x: (batch_num*batch_size*K+1*N, feat_dim*img_h*img_w)
        input_val = torch.cat([x, y_train], -1) #input: (batch_size*k+1*n, feat_dim+one_hot_dim=131)
    else: # labels_only
        input_val = y_train 
    
    input_val = input_val.to(device)
        
    # consolidate model input into torch_geometric Data object
    data = Data(x=input_val, y=y, edge_index=edge_index.t().contiguous(), train_mask=loss_mask)
    
    
    return data, paths