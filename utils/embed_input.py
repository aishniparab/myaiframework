import torch
from torch_geometric.data import Data

def embed_resnet(batch, num_flip, mask_type, edge_index, resnet, device, embedding_weight, img_h, img_w):
    '''
    - embedding_weight: multiply this value with the resnet output; if 0, the image information is 0
    '''
    x, y, paths = batch
    #x_shape: (batch_num, batch_size, k+1, n, img_h*img_w) 
    #y_shape: (batch_num, batch_size, k+1, n, one_hot_dim=3)
    #remove to use resnet x = x.to(device)
    edge_index = edge_index.to(device)
    
    #train_mask, y_train = get_train_mask(y) #train_mask: (batch_num*batch_size*K+1*N,)
    train_mask, y_train = get_flipped_loss_mask(y, num_flip, mask_type) # vary n to verify optimal decision rule should not be more than 70% when n=3
    train_mask = train_mask.to(device)
    y_train = y_train.to(device) # y_train is one hot with unknown label "0 0 1": (batch_num*batch_size*K+1*N, 3)
    
    #remove comment to turn on resnet x = x.view(-1, 1, img_h, img_w) #x: (batch_num*batch_size*K+1*N, channels=1, img_h, img_w)
    #print(train_mask, train_mask.shape, "train_mask")
    #print(y_train, y_train.shape, "y_train")
    
    """ # uncomment this to turn on resnet
    # disconnecting resnet
    x = resnet(x) #x: (batch_num*batch_size*K+1*N, feat_dim, img_h, img_w)
    x[x < 0] = 0 #hacky solution -- resnet should not output negative values bcos of relu
    x = x * embedding_weight # knob 
    x = x.squeeze() #x: (batch_num*batch_size*K+1*N, feat_dim*img_h*img_w)
    """

    y = y.view(-1, 3) #y: (batch_num*batch_size*K+1*N, one_hot_dim=3)
    y = y.to(device)
    

    # add noise or truth to resnet 
    # turining off resnet
    #input_val = torch.cat([x, y_train], -1) #input: (batch_num*batch_size*k+1*n, feat_dim+one_hot_dim=131)
    #input_val = input_val.to(device)
    input_val = y_train
    y = torch.argmax(y, dim=1) #y: (batch_num*batch_size*K+1*N,)
    
    data = Data(x=input_val, y=y, edge_index=edge_index.t().contiguous(), train_mask=train_mask)
    
    return data, paths