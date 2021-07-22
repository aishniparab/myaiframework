from utils.embed_input import embed
from utils.train import train
import torch

def sample_x_from_gaussian(mean_left, mean_right, std, batch_size, num_samples_per_class, num_classes, vector_dim):
	'''
	args:
	- mean_left: mean for left class
	- mean_right: mean for right class
	- std: standard deviation
	returns:
	- x values of shape (batch_size*num_samples_per_class*num_classes, vector_dim) sampled from gaussian
	'''
	assert num_classes == 2 # this function will only work if there are two classes
	sample_left = torch.normal(mean_left, std, size=(batch_size*num_samples_per_class, vector_dim))
	sample_right = torch.normal(mean_right, std, size=(batch_size*num_samples_per_class, vector_dim))
	sample = torch.stack([sample_left, sample_right], dim=1)
	sample = sample.view(-1, vector_dim)
	return sample

def test_optimal_decision_rule(tr_dataloader, model, loss_fn, optimizer, num_samples, edge_index, device, img_h, img_w, debug_step, embedding_weight, sample, vector_dim):
	print('===================================================')
	print("Train Mode: Running Optimal Decision Rule Test")
	print('===================================================')
	print("Measure loss on probe: you should see 100% accuracy\n",
		  "because model has no incentive.") 
	print('===================================================')
	for i in range(num_samples):
	  print("num flips = ", i)
	  batch = next(iter(tr_dataloader))
	  data, paths = embed(batch, i, True, edge_index, device, img_h, img_w, debug_step, embedding_weight, sample, vector_dim)
	  loss, acc, out, h = train(data, model, loss_fn, optimizer)
	print('===================================================')
	print("Measure loss on context and probe: you should see a\n",
		  "U shaped accuracy.") 
	print('===================================================')
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
	print('===================================================')
	for i in range(num_samples):
	  print("num flips = ", i)
	  batch = next(iter(tr_dataloader))
	  data, paths = embed(batch, i, False, edge_index, device, img_h, img_w, debug_step, embedding_weight, sample, vector_dim)
	  loss, acc, out, h = train(data, model, loss_fn, optimizer)
	for i in range(num_samples, -1, -1):
	  print("num flips = ", i)
	  batch = next(iter(tr_dataloader))
	  data, paths = embed(batch, i, False, edge_index, device, img_h, img_w, debug_step, embedding_weight, sample, vector_dim)
	  loss, acc, out, h = train(data, model, loss_fn, optimizer)
	print('===================================================')