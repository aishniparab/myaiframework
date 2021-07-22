import random
import numpy as np
import torch

class BatchSampler(object):
	'''
	Yields a batch of indexes at each iteration
	Usage: 
		tr_dataset = BongardDataset(batch_type='train', batch_size=8, one_hot_size=3, root='/Users/aish/Documents/deepmind/ShapeBongard_V2')

		tr_sampler = BatchSampler(labels=tr_dataset.y, batch_size=8)

		tr_dataloader = torch.utils.data.DataLoader(tr_dataset, sampler=tr_sampler)

		# for 2 epochs
		for epoch in range(2):
			tr_iter = iter(tr_dataloader)
			for batch in tqdm(tr_iter):
				x, y, paths = batch
				#print("x: ", x.shape, "y: ", y.shape)
				#print("paths: ")
				#print(paths)
	'''
	
	def __init__(self, random_seed, labels, batch_size):
		'''
		Initialize BatchSampler object
		Args:
		- labels: iterable containing labels for the dataset (sample indexes 
					will be infered from this iterable)
		- num_samples_per_class: number of samples perfor each iteration for each class
		'''
		super(BatchSampler, self).__init__()
		self.batch_size = batch_size
		self.Y = labels
		self.dataset_size = self.Y.__len__()
		self.num_batches = self.dataset_size/self.batch_size
		self.seed = random_seed


	def __len__(self):
		return self.num_batches

	def __iter__(self):

		#shuffle data before epoch
		indices = np.arange(self.dataset_size)
		random.seed(self.seed)
		np.random.shuffle(indices)
		
		for start in range(0, self.dataset_size, self.batch_size):
			end = min(start + self.batch_size, self.dataset_size)
			batch_idx = indices[start:end]
			yield batch_idx

