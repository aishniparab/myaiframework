import torch
import datetime

def compute_acc(model_output, labels):
    preds = model_output.argmax(dim=1)  # Use the class with highest probability.
    correct = int((preds == labels).sum())  # Check against ground-truth labels.
    #print("model_output: ", model_output, "preds: ", preds, "true: ", labels, "acc: ", correct/len(labels))
    acc = correct / len(labels)  # Derive ratio of correct predictions.
    return acc

def train(data, model, loss_fn, optimizer): 
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
    print("cross_entropy_loss: ", loss.item(), "acc: ", acc, "\n")
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