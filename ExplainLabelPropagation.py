from csv import excel_tab
from networkx.classes.function import neighbors
import numpy as np
from numpy.core.numeric import tensordot
from sklearn.preprocessing import normalize

from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

def build_weight_matrix(mention_graph, node_to_idx):
	N = len(node_to_idx)
	W = np.zeros((N, N))
	for u, neighbors in mention_graph.items():
		for v,weight in neighbors.items():
			try:
				i = node_to_idx[u]
				j = node_to_idx[v]
				W[i][j] = weight
			except KeyError:
				print("u: {} v: {}".format(u, v))
				exit(1)


	# check symmetry
	for i in range(N):
		for j in range(i):
			assert W[i][j] == W[j][i]
	return W

def build_transition_matrix(W):
	return normalize(W, axis=1, norm='l1')


class ExplainLabelPropagator:

	def __init__(self, mention_graph, num_of_labels, train_user_dict, max_iter=100):
		self.mention_graph = mention_graph
		self.nodes = mention_graph.keys()
		# build node dictionary
		self.node_to_idx = {node:i for i,node in enumerate(self.nodes)}
		# build transition matrix
		W = build_weight_matrix(mention_graph, self.node_to_idx)
		self.T = build_transition_matrix(W)
		# build labeling matrix
		num_of_nodes = len(self.nodes)
		self.num_of_nodes = num_of_nodes
		self.num_of_labels = num_of_labels
		self.train_user_dict = train_user_dict
		self.Y_true = {self.node_to_idx[node]:c for node,c in train_user_dict.items()}
		Y = (1.0/num_of_labels) * np.ones((num_of_nodes, num_of_labels)) # uniform
		self.Y = self.clamp_labeling_matrix(Y, self.Y_true)
		self.max_iter = max_iter
		# build feature importance matrix
		# num_of_train_nodes = len(train_user_dict)
		W = np.zeros((num_of_nodes, num_of_labels, num_of_nodes)) # zero
		self.W = self.clamp_feature_importance(W, self.Y_true)
		
		

	def clamp_labeling_matrix(self, Y, Y_true):
		#num_of_labels = Y.shape[1]
		for id,c in Y_true.items():
			dist = np.zeros(self.num_of_labels)
			dist[c] = 1.0
			Y[id] = dist
		return Y

	def clamp_feature_importance(self, W, Y_true):
		#num_of_train_nodes = len(Y_true)
		for id,c in Y_true.items():
			dist = np.zeros(self.num_of_nodes)
			dist[id] = 1.0
			W[id][c] = dist
		return W

	def labelprop(self, print_stats=False):
		print("Start label prop...")
		iter = 0
		for iter in range(self.max_iter):
			Y_prev = self.Y
			Y_next = np.matmul(self.T, self.Y)
			Y_next = self.clamp_labeling_matrix(Y_next, self.Y_true)
			W_next = np.tensordot(self.T, self.W, axes=1) # W[i,c,k] = sum_j T[i:j]*W[j,c,k]
			W_next = self.clamp_feature_importance(W_next, self.Y_true)
			self.Y = Y_next
			self.W = W_next
			if print_stats:
				test_users = set(self.nodes) - self.train_user_dict.keys()
				print("Iteration {} ---------------------------------------------".format(iter))
				Y_pred, W_pred = self.explain_by_most_influential_user(test_users)
				self.print_stats(test_users, Y_pred, W_pred)

			if np.allclose(Y_next, Y_prev):
				print("Converges!")
				break

		print("Finish label prop after {} iterations.".format(iter))

	def predict(self, test_users):
		idx = [self.node_to_idx[node] for node in test_users]
		Y_pred = np.argmax(self.Y, axis=-1)
		return Y_pred[np.array(idx)]

	def explain(self, test_users):
		idx = [self.node_to_idx[node] for node in test_users]
		Y_pred = np.argmax(self.Y, axis=-1)
		Y_pred = Y_pred[np.array(idx)]
		W_pred = self.W[np.array(idx), Y_pred]
		return Y_pred, W_pred
	
	def explain_by_most_influential_user(self, test_users):
		idx = [self.node_to_idx[node] for node in test_users]
		Y_pred = np.argmax(self.Y, axis=-1)
		Y_pred = Y_pred[np.array(idx)]
		W_pred = self.W[np.array(idx), Y_pred]
		W_pred = np.argmax(W_pred, axis=-1)
		return Y_pred, W_pred

	@staticmethod
	def print_stats(test_users, Y_pred, W_pred):
		#print("Start printing result...")
		for user, y, w in zip(test_users, Y_pred, W_pred):
			print("User: {} Y_pred: {} Most influence: {}".format(user, y, w))



###################################################################################################
# Experiment
####################################################################################

def test_case1():
	test_mention_graph = {
		'usr0': {'usr1': 2, 'usr2': 3},
		'usr1': {'usr0': 2, 'usr3': 4},
		'usr2': {'usr0': 3, 'usr3': 2, 'usr4': 1},
		'usr3': {'usr1': 4, 'usr2': 2, 'usr4': 2},
		'usr4': {'usr2': 1, 'usr3': 2}
	}
	num_of_labels = 2 # LA, SH
	train_user_dict = {'usr0':0, 'usr3':1}	
	model = ExplainLabelPropagator(test_mention_graph, num_of_labels, train_user_dict)
	print("nodes: ", model.node_to_idx)
	#print("Initial Y:", model.Y)
	#print("Initial W:", model.W)
	model.labelprop(print_stats=False)

def test_experiment1():
	# demo example from towards data science
	edges = [
		[1, 2],
		[1, 5],
		[2, 4],
		[4, 5],
		[3, 4],
		[4, 7],
		[5, 6],
		[6, 7],
		[7, 8],
		[4, 9],
		[9, 10],
		[10, 12],
		[10, 11],
		[9, 13],
		[9, 15],
		[13, 15],
		[15, 16],
		[13, 14]
	] 

	# build mention graph
	test_mention_graph = {}
	nodes = list(range(1, 17))
	for node in nodes:
		test_mention_graph[node] = {}
		
	for edge in edges:
		u, v = edge[0], edge[1]
		#weight = # uniform weight
		weight = np.random.randint(5)
		test_mention_graph[u][v] = weight
		test_mention_graph[v][u] = weight

	# plot
	G = nx.Graph()
	G.add_node(1, label='LA')
	G.add_node(8, label="LA")
	G.add_node(14, label="SH")
	G.add_node(16, label="SH")
	for u, neighbors in test_mention_graph.items():
		for v, weight in neighbors.items():
			if u < v:
				G.add_edge(u, v, weight=weight)

	nx.draw_networkx(G, with_labels=True)
	plt.show()

	# add true labels
	num_of_labels = 2 # 0 for LA and 1 for SH
	train_user_dict = {1:0, 8:0, 14:1, 16:1}
	test_users = list(set(nodes) - train_user_dict.keys())
	model = ExplainLabelPropagator(test_mention_graph, num_of_labels, train_user_dict)
	print("nodes: ", model.node_to_idx)
	#print("Initial Y:", model.Y)
	#print("Initial W:", model.W)
	model.labelprop(print_stats=True)
	print("Final result: ")
	Y_pred, W_pred = model.explain(test_users)
	W_pred = normalize(W_pred,axis=1,norm='l1')
	#Y_pred, W_pred = model.explain_by_most_influential_user(test_users)
	model.print_stats(test_users, Y_pred, W_pred)





if __name__ == "__main__":
	np.random.seed(1024)
	#test_case1()
	test_experiment1()