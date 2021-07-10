import torch


def get_flipped_loss_mask(y, n=0, mask_type="probe"):
  '''
  same as get_loss_mask function with labels of first n samples flipped without replacement;
  args:
  - y: list of targets shape (batch_num=1, batch_size=10, K=7, N=2, size_of_one_hots=3)
  - n: number of digits to flip
  return:
  - loss_mask: bool list indicating True for samples to compute loss over (not same as GNN loss_mask)
  - y_train: labels with probe ones masked as "unknown" and first n flipped
  '''
  y_train = y.clone() # to flip labels

  # if n>0 flip n labels, else y_train is unchanged
  if n > 0 and n < 7:
    # create mask to flip labels based on input
    loss_mask = torch.as_tensor([[[-1, 1, 0], [1, -1, 0]] for i in range(n)], dtype=torch.float32)
    
    # apply mask to first n examples
    for sample in y_train[:, :, :n]:
      sample+=loss_mask 
  
  if mask_type == "probe":
    is_probe_mask = y_train.clone()
    is_probe_mask[:, :, -1] += torch.as_tensor([[-1, 0, 0], [0, -1, 0]], dtype=torch.float32)
    is_probe_mask = is_probe_mask.view(-1, 3)
    is_context_mask = torch.any(is_probe_mask, dim=1)
    loss_mask = ~is_context_mask
  else:
    loss_mask = torch.ones(y_train.view(-1, 3).shape[0], dtype=torch.bool) # no masking for now see line 95
  
  y_train = y_train.view(-1, 3) # (1*6*7*2, 3) = (84, 3) # for output to be (84,3) # dont view (-1, 3) ? forgot why
  
  return loss_mask, y_train


def get_train_mask(y):
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
