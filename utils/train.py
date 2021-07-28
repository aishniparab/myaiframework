import torch
from torch import nn
import datetime

def reset_weights(model):
  # refactor to accept any model and reset all of its parameters
  model.linear_layer.reset_parameters()

def get_loss_mask(y, num_flips=0, mask_type=True):
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
  if num_flips > 0 and num_flips <= 7:
    # create mask to flip labels based on input
    loss_mask = torch.as_tensor([[[-1, 1, 0], [1, -1, 0]] for i in range(num_flips)], dtype=torch.float32)
    
    # apply mask to first n examples
    for sample in y_train[:, :, :num_flips]:
      sample+=loss_mask 
  
  if mask_type: # mask context
    is_probe_mask = y_train.clone()
    is_probe_mask[:, :, -1] += torch.as_tensor([[-1, 0, 0], [0, -1, 0]], dtype=torch.float32)
    is_probe_mask = is_probe_mask.view(-1, 3)
    is_context_mask = torch.any(is_probe_mask, dim=1)
    loss_mask = ~is_context_mask
  else:
    loss_mask = torch.ones(y_train.view(-1, 3).shape[0], dtype=torch.bool) # no masking context for now see line 95
  
  y_train = y_train.view(-1, 3) # (1*6*7*2, 3) = (84, 3) # for output to be (84,3) # dont view (-1, 3) ? forgot why
  
  return loss_mask, y_train

def compute_acc(model_output, labels):
    preds = model_output.argmax(dim=1)  # Use the class with highest probability.
    correct = int((preds == labels).sum())  # Check against ground-truth labels.
    #print("model_output: ", model_output, "preds: ", preds, "true: ", labels, "acc: ", correct/len(labels))
    acc = correct / len(labels)  # Derive ratio of correct predictions.
    return acc

def train(data, model, loss_fn, optimizer): # refactor
    out, h = model(data.x.float(), data.edge_index.view(2, -1)) #num_edges))
    probe_preds = out[data.train_mask].view(-1, 2) #(batch_size, 2) #assumes out is single pred
    probe_y_right = data.y[data.train_mask].view(-1, 2)[:, 1] #(batch_size,) 
    assert probe_y_right.sum() == len(probe_y_right) #if all ones then sum = len
    assert probe_preds.shape[0] == probe_y_right.shape[0] 
    loss = loss_fn(probe_preds, probe_y_right)
    optimizer.zero_grad()  # Clear gradients.
    loss.backward(retain_graph=True)  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    acc = compute_acc(probe_preds, probe_y_right)
    #print("cross_entropy_loss: ", loss.item(), "acc: ", acc, "\n")
    return loss, acc, out, h

def val(data, model, loss_fn):
    out, h = model(data.x.float(), data.edge_index.view(2, -1)) #num_edges))  
    probe_preds = out[data.train_mask].view(-1, 2) #(batch_size, 2) #assumes out is single pred
    probe_y_right = data.y[data.train_mask].view(-1, 2)[:, 1] #(batch_size,) 
    assert probe_y_right.sum() == len(probe_y_right) #if all ones then sum = len
    assert probe_preds.shape[0] == probe_y_right.shape[0] 
    
    loss = loss_fn(probe_preds, probe_y_right)

    acc = compute_acc(probe_preds, probe_y_right)
    
    return loss, acc, out, h

