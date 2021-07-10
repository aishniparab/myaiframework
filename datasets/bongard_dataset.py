import os
import numpy as np
import random
import torch
import torch.utils.data as data
from tqdm import tqdm 
from torchvision import transforms

class BongardDataset(data.Dataset):
	"""
		https://github.com/NVlabs/Bongard-LOGO
	"""

	def __init__(self, random_seed=123, batch_type='train', img_dim=(512,512), batch_size=None, one_hot_size=3, root='./ShapeBongard_V2'):
		'''
		Args:
		- batch_type: training, testing or validation set
		- img_dim: (height, weight) of image in input layer
		- root: directory where dataset will be stored
		- one_hot_size: one_hot_vector size of a label (left, right, unlabeled)
		Usage: 
			tr_dataset = BongardDataset(batch_type='train', one_hot_size=3, root='./ShapeBongard_V2')
			# returns tr_dataset.y, tr_dataset.x_paths
		'''

		super(BongardDataset, self).__init__()
		self.seed = random_seed
		self.root = root
		self.batch_type = batch_type
		self.batch_size = batch_size
		self.one_hot_size = one_hot_size
		self.img_h, self.img_w = img_dim
		self.img_dim = self.img_h*self.img_w
		
		# as stated in paper
		self.num_train = 9300
		self.num_val = 900

		# in dataset dir
		self.num_classes = 2
		self.num_samples_per_class = 7
		

		# resize original 512x512 image to 256x246
		self.transform = transforms.Compose([transforms.ToPILImage(mode=None),
												transforms.Resize(img_dim)])

		if not os.path.exists(self.root):
			raise RuntimeError('Dataset not found.')

		
		# basic, free-form, abstract --> images --> pos, neg --> img.png
		# problem_type/images/problem_class/img.png
		problem_folders = [os.path.join(self.root, problem_type, 'images', problem_class) #img path
						  for problem_type in os.listdir(self.root) # basic, free-form, abstract
						  if os.path.isdir(os.path.join(self.root, problem_type))
						  for problem_class in os.listdir(os.path.join(self.root, problem_type, 'images')) # neg, pos
						  if os.path.isdir(os.path.join(self.root, problem_type, 'images', problem_class))]

		random.seed(self.seed)
		random.shuffle(problem_folders)

		if self.batch_type == 'train':
			self.folders = problem_folders[: self.num_train]
		elif self.batch_type == 'val':
			self.folders = problem_folders[self.num_train : self.num_train + self.num_val]
		elif self.batch_type == 'test':
			self.folders = problem_folders[self.num_train + self.num_val:]
		else:
			raise ValueError('Batch must be of type Train, Validation or Test')

		get_label = lambda folder, class_name : [class_name for problem_img in 
						os.listdir(os.path.join(folder, str(class_name)))]
		print("Fetching Y"+batch_type+" labels")
		self.y = np.array([list(zip(np.eye(one_hot_size)[get_label(problem, 0)], 
					np.eye(one_hot_size)[get_label(problem, 1)])) 
					for problem in tqdm(self.folders)])
		assert self.y.shape == (len(self.folders), self.num_samples_per_class, 
								self.num_classes, self.one_hot_size)

		print("Fetching X"+batch_type+" paths")		
		get_img_path = lambda folder, class_name: [os.path.join(folder, str(class_name), problem_img) 
							for problem_img in os.listdir(os.path.join(folder, str(class_name)))]
		self.x_paths = np.array([list(zip(get_img_path(problem, 0), get_img_path(problem, 1))) 
							for problem in tqdm(self.folders)])
		assert self.x_paths.shape == (len(self.folders), self.num_samples_per_class, self.num_classes)

	def __getitem__(self, idx):
		'''
		Args:
		- idx: problem at idx
		
		Returns:
		- problem_imgs: img data for each img in problem, shape: num_samples_per_class x img_dim
		- labels: labels for each img in problem, shape: num_samples_per_class x num_classes x one_hot_size
		- problem_path: str obj is path to problem
		'''
		
		get_imgs_at_idx = lambda x: torch.stack([torch.stack([image_file_to_array(class_1, self.transform), 
										image_file_to_array(class_2, self.transform)])
										for class_1, class_2 in x])

		batch_imgs = torch.stack([get_imgs_at_idx(batch_i) for batch_i in self.x_paths[idx]])

		#assert batch_imgs.shape == (self.batch_size, self.num_samples_per_class, 
		#							self.num_classes, self.img_dim)
		
		batch_y = torch.from_numpy(self.y[idx])
		
		batch_path = [os.path.split(path)[0] for path in self.x_paths[idx][:, 1, 1]]
		
		return batch_imgs, self.y[idx], batch_path

	def __len__(self):
		return len(self.x_paths)

#class BongardLOGODataset(data.Dataset):
"""
LOGO vectors for program induction
"""

