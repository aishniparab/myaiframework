from scipy.linalg import null_space
from scipy.sparse import csgraph, linalg
import torch
import numpy as np
import networkx as nx

class Graph(object):
	def __init__(self, dataset_name, img_dim, batch_size, num_classes, num_context_per_class, num_probe):	
		self.dataset_name = dataset_name
		self.img_dim = img_dim
		self.batch_size = batch_size
		self.num_classes = num_classes #N
		self.num_context_per_class = num_context_per_class #K
		self.num_probe = num_probe #1
		
		# for a single graph G
		self.num_nodes_per_graph = (self.num_context_per_class + self.num_probe) * self.num_classes #(K+1)*2
		self.num_edges_per_graph = num_nodes_per_graph * (num_nodes_per_graph - 1)
		self.send_idx_per_graph = np.arange(num_nodes_per_graph)
		self.receive_idx_per_graph = np.arange(num_nodes_per_graph)
		self.edge_index_per_graph = torch.tensor([[x,y] for x in send_idx_per_graph 
		                          for y in receive_idx_per_graph 
		                          if x != y]).t().contiguous() # 14 * 13 pairs
		self.G = nx.Graph()
		self.G.add_edges_from(edge_index_per_graph.t().numpy())
		self.num_connected_components = null_space(nx.laplacian_matrix(G, nodelist=np.arange(num_nodes_per_graph)).A).shape[1]
		
		# for batch of graphs G * batch_size
		self.num_graphs = batch_size
		self.num_nodes = num_graphs * num_nodes_per_graph
		self.num_edges =  num_graphs * num_edges_per_graph
		self.send_idx = np.split(np.arange(num_nodes), num_graphs)
		self.receive_idx = np.split(np.arange(num_nodes), num_graphs)
		self.edge_index = torch.tensor([[x,y] for i in range(num_graphs)
		                                for x in send_idx[i] 
		                                for y in receive_idx[i]
		                          if x != y]).t().contiguous()       #torch.block_diag(edge_index_per_graph.t(), edge_index_per_graph.t(), dim=)     # stack edge_index_per_graph
		self.G_prime = nx.Graph()
		self.G_prime.add_edges_from(edge_index.t().numpy())
		self.num_conntected_components = null_space(nx.laplacian_matrix(G_prime, nodelist=np.arange(num_nodes)).A).shape[1])
	
	def get_num_connected_components(graph):
		return null_space(nx.laplacian_matrix(graph, nodelist=np.arange(num_nodes)).A).shape[1])
		
	def draw(save_plot=False, save_path, dpi)__:
		nx.draw(self.G_prime, node_size=self.batch_size*0.5, width=0.1, pos=nx.spring_layout(self.G_prime, k=1/self.batch_size*4, scale=0.01, seed=1))
		if save_plot:
			plt.savefig(save_path, dpi=dpi, bbox_inches = 'tight')

	def print_info():
		print("\n")
		print('===================================================')
		print("Dataset: %s".format(dataset_name))
		print('===================================================')
		print("num_classes: ", num_classes)
		print("num_samples_per_class: ", num_context_per_class)
		print("batch_size: ", batch_size)
		print("img_width, img_height: ", img_dim)
		print('===================================================')
		print("\n")	
		print('===================================================')
		print("Single Graph")
		print('===================================================')
		print("num_nodes_per_graph: ", num_nodes_per_graph)
		print("num_edges_per_graph: ", num_edges_per_graph)
		print("edge_index_per_graph_shape: ", edge_index_per_graph.shape)
		#print("nx_laplacian_matrix(): ", nx.laplacian_matrix(G, nodelist=np.arange(num_nodes_per_graph)))
		#print("num_of_0_eigen_values: ", linalg.eigs(nx.laplacian_matrix(G, nodelist=np.arange(num_nodes_per_graph)).A, sigma=0, return_eigenvectors=False))
		print("num_connected_components: ", get_num_connected_components(self.G))
		print("num_connected_components == 1: ", get_num_connected_components(self.G) == 1)
		print('===================================================')
		print("\n")
		print('===================================================')
		print("All Graphs")
		print('===================================================')
		print("num_nodes: ", num_nodes)
		print("num_edges: ", num_edges)
		print("edge_index_shape: ", edge_index.shape)
		#print("nx_laplacian_matrix(): ", nx.laplacian_matrix(G_prime, nodelist=np.arange(num_nodes)))
		#print("num_of_0_eigen_values: ", linalg.eigs(nx.laplacian_matrix(G_prime, nodelist=np.arange(num_nodes)).A, sigma=0, return_eigenvectors=False))
		print("num_connected_components: ", get_num_connected_components(self.G_prime))
		print("num_connected_componenets == batch_size: ", get_num_connected_components(self.G_prime) == batch_size)
		print('===================================================')
		print("\n")



