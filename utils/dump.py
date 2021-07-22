# old implementations

def get_loss_mask(y): #orig function without flips
  # refacor
  ''' 
  train_mask of a graph data object describes for which nodes we already know the labels of 
  args:
  - y: list of targets shape (batch_num=1, batch_size=6, K=7, N=2, size_of_one_hots=3)
  return:
  - train_mask: bool list indicating False for nodes we want to hide the true values of
  - y_train: labels with probe ones masked as "unknown"
  '''
  y_train = y.clone()
  # create probe mask
  mask = torch.as_tensor([[-1, 0, 1], [0, -1, 1]], dtype=torch.float32)
  
  # apply mask
  y_train[:, :, -1] += mask # use mask to hide true labels of last targets # (1, 6, 7, 2, 3)
  y_train = y_train.view(-1, 3) # (1*6*7*2, 3) = (84, 3)
  # dont view (-1, 3)
  # mask out the context ones
  
  # refactor below
  train_mask = torch.argmax(y_train, dim=1) # (84,)  
  train_mask[train_mask<2] = False
  train_mask[train_mask>1] = True # sets probes to true
  train_mask = train_mask.bool() # (84,)
  
  return train_mask, y_train #input labels


  """
  print("num_flips = 0, correct_pred/total = 7/7 = 1.0", "\n")
  print("num_flips = 1, correct_pred/total = 6/7 = 0.857", "\n")
  print("num_flips = 2, correct_pred/total = 5/7 = 0.714", "\n")
  print("num_flips = 3, correct_pred/total = 4/7 = 0.571", "\n")
  print("num_flips = 4, correct_pred/total = 3/7 = 0.429", "\n")
  print("num_flips = 5, correct_pred/total = 2/7 = 0.286", "\n")
  print("num_flips = 6, correct_pred/total = 1/7 = 0.143", "\n")
  print("num_flips = 7, correct_pred/total = 0/7 = 0.000", "\n")
  print("num_flips = 6, correct_pred/total = 1/7 = 0.143", "\n")
  print("num_flips = 5, correct_pred/total = 2/7 = 0.286", "\n")
  print("num_flips = 4, correct_pred/total = 3/7 = 0.429", "\n")
  print("num_flips = 3, correct_pred/total = 4/7 = 0.571", "\n")
  print("num_flips = 2, correct_pred/total = 5/7 = 0.714", "\n")
  print("num_flips = 1, correct_pred/total = 6/7 = 0.857", "\n")
  print("num_flips = 0, correct_pred/total = 7/7 = 1.0", "\n")
  """

  def embed_flips(batch, num_flip, mask_type, edge_index, device, img_h, img_w):
    '''
    args:
    - batch.x.shape: (batch_num, batch_size, k+1, n, img_h*img_w) 
    - batch.y.shape: (batch_num, batch_size, k+1, n, one_hot_dim=3)
    - num_flip: number of labels to flip
    - mask_probe: bool indicates whether probe should be masked for loss computation

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
    
    # pass y as input to model
    input_val = y_train 
    
    # consolidate model input into torch_geometric Data object
    data = Data(x=input_val, y=y, edge_index=edge_index.t().contiguous(), train_mask=loss_mask)
    
    return data, paths

  