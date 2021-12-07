# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 15:29:45 2021
"""
#%%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import data
from preprocess import mention_graph, data_rows
from accuracy import mean_absolute, median_absolute, mean_mse_2d, median_mse_2d, mean_mse_3d, median_mse_3d
import pandas as pd
import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# https://www.geeksforgeeks.org/visualize-graphs-in-python/

#%%
class GraphVisualization:
   
    def __init__(self):
        self.visual = []
          
    def addEdge(self, a, b):
        temp = [a, b]
        self.visual.append(temp)
          
    def visualize(self):
        G = nx.Graph()
        print("graph is ok")
        G.add_edges_from(self.visual)
        print("adding edges")
        pos = nx.spring_layout(G, k=0.05, iterations=20)
        colorlist = [ 'r', 'g', 'b', 'c', 'm', 'y', 'k' ]
        # pos = nx.kamada_kawai_layout(G)
        # nx.draw_networkx(G, pos, with_labels=False, node_size=3, edge_color = colorlist[random.randint(0,6)], width = 0.5)
        nx.draw_networkx(G, pos, with_labels=False, node_size=3, edge_color='skyblue', width = 0.5)
        print("drawing finished")
        plt.figure(figsize=(200,200), dpi=1000)
        plt.show()
        print("show the plot")
        
if __name__ == "__main__":
    G = GraphVisualization()
    print("creating a graph")
    for i in range(len(data_rows)):
        if len(mention_graph[data_rows[i]['user_id']]) < 9:
            continue
        for j in mention_graph[data_rows[i]['user_id']].keys():
            G.addEdge(data_rows[i]['user_id'], j)
    # df = pd.DataFrame(index=G.nodes(), columns=G.nodes())
    # for row, data in nx.shortest_path_length(G):
    #     for col, dist in data.items():
    #         df.loc[row,col] = dist

    # df = df.fillna(df.max().max())
    print("finish adding edges")
    G.visualize()


# %%
G = nx.Graph()
colorlist = [ 'r', 'g', 'b', 'c', 'm', 'y', 'k' ]
for i in range(len(data_rows)):
    if len(mention_graph[data_rows[i]['user_id']]) < 10:
        continue
    for j in mention_graph[data_rows[i]['user_id']].keys():
        G.add_edge(data_rows[i]['user_id'], j, color=colorlist[random.randint(0, 6)])
colors = nx.get_edge_attributes(G,'color').values()
pos = nx.spring_layout(G, k=0.05, iterations=20)
nx.draw_networkx(G, pos, with_labels=False, edge_color=colors, node_size=3, width = 0.5)
plt.savefig("10mentions.png")
plt.show()


# %%
n = 3
# X = ['distance_diff', 'mse_2d', 'mse_3d']
X = np.arange(n)+1
plt.figure(figsize=(9,6))
Y1 = [0.74, 0.826, 0.817]
Y2 = [0.4098, 0.596, 0.5517]
plt.bar(X, Y1, width = 0.3, facecolor = 'royalblue', label='median', lw=1)
plt.bar(X+0.3, Y2, width = 0.3, facecolor = 'cornflowerblue', label='mean', lw=1)
plt.legend(loc="upper left")
plt.xticks(X+0.15, ['Distance Difference', 'MSE(2D)', 'MSE(3D)'])
plt.xlabel("Methods")
plt.ylabel("Accuracy Percentage")
plt.title("Accuracy with 8:2 train/ test ratio and 5 iterations")
# names = ['distance_diff', 'mse_2d', 'mse_3d']
# values = [mean_absolute, median_absolute, mean_mse_2d, median_mse_2d, mean_mse_3d, median_mse_3d]

# plt.bar(names, values)
# plt.suptitle('Accuracy')
plt.savefig("80per_5iter.png")
plt.show()



# %% 5 iterations
n = 5
X = ['50%', '60%', '70%', '80%', '90%']
plt.figure(figsize=(25,25))
# fig, axs(ax1, ax2) = plt.subplots(2)
fig, axs = plt.subplots(2, figsize=(15,15))

Y1 = [0.74688, 0.74633, 0.745, 0.74, 0.7556]
Y2 = [0.4001556, 0.4118, 0.418747, 0.4098, 0.43725]
Y3 = [0.83584, 0.835996, 0.8388, 0.826, 0.8441215]
Y4 = [0.58687, 0.5937398, 0.6005256, 0.596, 0.6155878]
Y5 = [0.82417, 0.82328, 0.82742, 0.817, 0.8361955]
Y6 = [0.539419, 0.55, 0.558, 0.5517, 0.5693527]

axs[0].set_title('Median', fontsize = 20)
axs[1].set_title('Mean', fontsize = 20)
axs[0].scatter(X, Y1, s=20)
axs[1].scatter(X, Y2, s=20)
axs[0].scatter(X, Y3, s=20)
axs[1].scatter(X, Y4, s=20)
axs[0].scatter(X, Y5, s=20)
axs[1].scatter(X, Y6, s=20)

# axs[0].label_outer()
axs[0].set_yticks(np.arange(0.736, 0.85, step=0.008))
axs[1].set_yticks(np.arange(0.395, 0.65, step=0.01))

axs[0].plot(X, Y1, label='Distance Difference')
axs[1].plot(X, Y2, label='Distance Difference')
axs[0].plot(X, Y3, label='MSE(Lat&Lon)')
axs[1].plot(X, Y4, label='MSE(Lat&Lon)')
axs[0].plot(X, Y5, label='MSE(Cartesian)')
axs[1].plot(X, Y6, label='MSE(Cartesian)')
# plt.xticks(X, ['50%', '60%', '70%', '80%', '90%'], fontsize = 15)
axs[1].legend(loc="right", prop={'size': 10})
axs[0].legend(loc="right", prop={'size': 10})
plt.xlabel("train/test ratio", fontsize = 20)
axs[1].set_ylabel("Accuracy", fontsize = 20)
axs[0].set_ylabel("Accuracy", fontsize = 20)
plt.suptitle("Accuracy VS different train/ test ratio with 5 iterations", fontsize = 20)
plt.savefig("ratio_comparison_5iter.png")


# %% 6 iterations
n = 5
X = np.arange(n)+1
# plt.figure(figsize=(10,10))

Y1 = [0.78475, 0.77828, 0.7696, 0.75995, 0.76486]
Y2 = [0.86307, 0.86012, 0.856329, 0.83952, 0.85]
Y3 = [0.852178, 0.848386, 0.84494, 0.83024, 0.8428]

plt.scatter(X, Y1, s=10)
plt.scatter(X, Y2, s=10)
plt.scatter(X, Y3, s=10)

plt.plot(X, Y1, label='Distance Difference')
plt.plot(X, Y2, label='MSE(Lat&Lon)')
plt.plot(X, Y3, label='MSE(Cartesian)')

plt.xticks(X, ['50% train data', '60% train data', '70% train data', '80% train data', '90% train data'])
plt.legend(loc="right", prop={'size': 8})

plt.xlabel("train/test ratio")
plt.ylabel("Accuracy")

plt.title("Accuracy VS different train/ test ratio with 6 iterations")
plt.savefig("ratio_comparison_6iter.png")


# %% 8:2 train/test ratio
n = 10
X = np.arange(n)+1
# plt.figure(figsize=(15,25))
# fig, axs(ax1, ax2) = plt.subplots(2)
fig, axs = plt.subplots(2, figsize=(8,10))

Y1 = [0.7374, 0.76, 0.74, 0.7599, 0.74, 0.75995, 0.75995, 0.75995, 0.75995, 0.75995]
Y2 = [0.4986737, 0.455, 0.428, 0.4244, 0.4098, 0.41578, 0.40716, 0.41313, 0.40318, 0.4098]
Y3 = [0.827586, 0.839, 0.826, 0.8395, 0.826, 0.83952, 0.83952, 0.83952, 0.83952, 0.83952]
Y4 = [0.669, 0.637, 0.6127, 0.608, 0.596, 0.59482, 0.5855, 0.5915, 0.58687, 0.59]
Y5 = [0.817639, 0.83, 0.816, 0.830, 0.817, 0.83024, 0.83024, 0.83024, 0.83024, 0.83024]
Y6 = [0.628647, 0.594, 0.5669, 0.5656, 0.5517, 0.553, 0.547, 0.54907, 0.5444297, 0.5491]

axs[0].set_title('Median')
axs[1].set_title('Mean')
axs[0].scatter(X, Y1, s=20)
axs[1].scatter(X, Y2, s=20)
axs[0].scatter(X, Y3, s=20)
axs[1].scatter(X, Y4, s=20)
axs[0].scatter(X, Y5, s=20)
axs[1].scatter(X, Y6, s=20)

axs[0].label_outer()
axs[0].set_yticks(np.arange(0.736, 0.85, step=0.008))
axs[1].set_yticks(np.arange(0.395, 0.7, step=0.02))

axs[0].plot(X, Y1, label='Distance Difference')
axs[1].plot(X, Y2, label='Distance Difference')
axs[0].plot(X, Y3, label='MSE(Lat&Lon)')
axs[1].plot(X, Y4, label='MSE(Lat&Lon)')
axs[0].plot(X, Y5, label='MSE(Cartesian)')
axs[1].plot(X, Y6, label='MSE(Cartesian)')
plt.xticks(X, ['1', '2', '3', '4','5', '6 ', '7', '8', '9', '10'])
axs[1].legend(loc="upper right", prop={'size': 8})
axs[0].legend(loc="right", prop={'size': 8})
plt.xlabel("Number of Iterations")
axs[1].set_ylabel("Accuracy")
axs[0].set_ylabel("Accuracy")
plt.suptitle("Accuracy VS different iterations with 8:2 train/ test ratio")
plt.savefig("iter_comparison_82.png")
# %%
