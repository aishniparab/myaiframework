import yaml
import os
import argparse
import random
import torch
from torch import nn
from samplers.batch_sampler import BatchSampler
from datasets.bongard_dataset import BongardDataset
from models.linear import Linear
from representation.graph.init import Graph
from utils.debug import test_optimal_decision_rule, sample_x_from_gaussian

def main(config):
	#yaml.dump(config, open(os.path.join(save_dir, 'config.yaml'), 'w'))

	# init tensorboard
	# from tensorboardX import SummaryWriter 
	# writer = SummaryWriter(os.oath.join(save_path, 'tensorboard'))
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	### Dataset ###
	num_classes, num_context_per_class = config['num_classes'], config['num_context_per_class']
	num_probe_per_class = config['num_probe_per_class']
	
	dataset_name = config['dataset']
	dataset_dir = config['dataset_args']['path']
	
	batch_size = config['batch_size']
	one_hot_size = config['dataset_args']['one_hot_size']

	img_h, img_w = config['dataset_args']['img_h'], config['dataset_args']['img_w']
	img_dim = (img_h, img_w)
	
	seed = config['seed']

	tr_dataset = BongardDataset(random_seed=seed, batch_type='train', img_dim=img_dim, batch_size=batch_size, one_hot_size=one_hot_size, root=dataset_dir)
	val_dataset = BongardDataset(random_seed=seed, batch_type='val', img_dim=img_dim, batch_size=batch_size, one_hot_size=one_hot_size, root=dataset_dir)
	test_dataset = BongardDataset(random_seed=seed, batch_type='test', img_dim=img_dim, batch_size=batch_size, one_hot_size=one_hot_size, root=dataset_dir)

	tr_sampler = BatchSampler(random_seed=seed, labels=tr_dataset.y, batch_size=batch_size)
	val_sampler = BatchSampler(random_seed=seed, labels=val_dataset.y, batch_size=batch_size)
	test_sampler = BatchSampler(random_seed=seed, labels=test_dataset.y, batch_size=batch_size)

	tr_dataloader = torch.utils.data.DataLoader(tr_dataset, sampler=tr_sampler, drop_last=config['dataset_args']['drop_last'])
	val_dataloader = torch.utils.data.DataLoader(val_dataset, sampler=val_sampler, drop_last=config['dataset_args']['drop_last'])
	test_dataloader = torch.utils.data.DataLoader(test_dataset, sampler=test_sampler, drop_last=config['dataset_args']['drop_last'])
	
	### Graph ###
	graph = Graph(dataset_name, img_dim, batch_size, num_classes, num_context_per_class, num_probe_per_class)
	graph.print_info(show_plot=False)
	
	### Model and Optimizer ###
	print('===================================================')
	# could add config item to load model from file
	# could refactor how you run tests with different model
	if config['model'] == 'linear':
		model = Linear(random_seed=seed, in_dim=config['model_args']['in_dim'], out_dim=config['model_args']['out_dim'])
		print("Model: ", model, "\n")

	if config['optimizer'] == 'adam':
		print("Optimizer: ", config['optimizer'], "\n")
		optimizer = torch.optim.Adam(model.parameters(), lr=config['optimizer_args']['lr'], weight_decay=config['optimizer_args']['weight_decay'])

	if config['loss_fn'] == 'cross_entropy':
		print("Loss: ", config['loss_fn'])
		loss_fn = nn.CrossEntropyLoss()
	print('===================================================')
	print("\n")
	
	if config['debug_mode']:
		num_samples_per_class = num_context_per_class + num_probe_per_class
		#test_optimal_decision_rule(tr_dataloader, model, loss_fn, optimizer, num_samples_per_class, graph.edge_index, device, img_h, img_w, "labels_only", None, None, None)
		#test_optimal_decision_rule(tr_dataloader, model, loss_fn, optimizer, num_samples_per_class, graph.edge_index, device, img_h, img_w, "resnet_labels", 1, None, None)
		
		sample = sample_x_from_gaussian(config['debug_args']['gaussian_args']['mean_left'], 
														config['debug_args']['gaussian_args']['mean_right'], 
														config['debug_args']['gaussian_args']['std'], 
														batch_size, num_samples_per_class, num_classes, 
														config['debug_args']['gaussian_args']['vector_dim'])

		#test_optimal_decision_rule(tr_dataloader, model, loss_fn, optimizer, num_samples_per_class, graph.edge_index, device, img_h, img_w, "gaussian", None, sample, config['debug_args']['gaussian_args']['vector_dim'])
		#doesnt work with resnet 
		test_optimal_decision_rule(tr_dataloader, model, loss_fn, optimizer, num_samples_per_class, graph.edge_index, device, img_h, img_w, "resnet_gaussian_labels", 1, sample, config['debug_args']['gaussian_args']['vector_dim'])

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config')
	parser.add_argument('--experiment_tag')
	parser.add_argument('--save_dir', default='./save')
	args = parser.parse_args()

	config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

	main(config)