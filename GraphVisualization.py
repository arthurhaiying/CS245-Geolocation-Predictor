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
    if len(mention_graph[data_rows[i]['user_id']]) < 8:
        continue
    for j in mention_graph[data_rows[i]['user_id']].keys():
        G.add_edge(data_rows[i]['user_id'], j, color=colorlist[random.randint(0, 6)])
colors = nx.get_edge_attributes(G,'color').values()
pos = nx.spring_layout(G, k=0.05, iterations=20)
nx.draw_networkx(G, pos, with_labels=False, edge_color=colors, node_size=3, width = 0.5)
plt.savefig("8mentions.png", dpi=1000)
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
plt.savefig("80per_5iter.png", dpi=1000)
plt.show()



# %%
n = 3
X = np.arange(n)+1
plt.figure(figsize=(9,10))
# fig, axs(ax1, ax2) = plt.subplots(2)
fig, axs = plt.subplots(2)

Y1 = [0.745, 0.8388, 0.82742]
Y2 = [0.418747, 0.6005256, 0.558]
Y3 = [0.74, 0.826, 0.817]
Y4 = [0.4098, 0.596, 0.5517]
Y5 = [0.7556, 0.8441215, 0.8361955]
Y6 = [0.43725, 0.6155878, 0.5693527]

axs[0].set_title('Median')
axs[1].set_title('Mean')
axs[0].scatter(X, Y1, s=10)
axs[1].scatter(X, Y2, s=10)
axs[0].scatter(X, Y3, s=10)
axs[1].scatter(X, Y4, s=10)
axs[0].scatter(X, Y5, s=10)
axs[1].scatter(X, Y6, s=10)

axs[0].label_outer()

axs[0].plot(X, Y1, label='70% train data')
axs[1].plot(X, Y2, label='70% train data')
axs[0].plot(X, Y3, label='80% train data')
axs[1].plot(X, Y4, label='80% train data')
axs[0].plot(X, Y5, label='90% train data')
axs[1].plot(X, Y6, label='90% train data')
plt.xticks(X, ['Distance Difference', 'MSE(2D)', 'MSE(3D)'])
axs[1].legend(loc="upper left")
axs[0].legend(loc="upper left")
plt.xlabel("Methods")
axs[1].set_ylabel("Percentage")
axs[0].set_ylabel("Percentage")
plt.suptitle("Accuracy VS different train/ test ratio with 5 iterations")
plt.savefig("per_comparison_5iter.png", dpi=1000)


# %%
n = 3
X = np.arange(n)+1
plt.figure(figsize=(9,10))
# fig, axs(ax1, ax2) = plt.subplots(2)
fig, axs = plt.subplots(2)

Y1 = [0.74, 0.826, 0.817]
Y2 = [0.4098, 0.596, 0.5517]
Y3 = [0.75995, 0.83952, 0.83024]
Y4 = [0.41313, 0.5915, 0.54907]
Y5 = [0.7599469, 0.8395225, 0.8302387]
Y6 = [0.4098, 0.59, 0.5491]

axs[0].set_title('Median')
axs[1].set_title('Mean')
axs[0].scatter(X, Y1, s=10)
axs[1].scatter(X, Y2, s=10)
axs[0].scatter(X, Y3, s=10)
axs[1].scatter(X, Y4, s=10)
axs[0].scatter(X, Y5, s=10)
axs[1].scatter(X, Y6, s=10)

axs[0].label_outer()

axs[0].plot(X, Y1, label='5 iterations')
axs[1].plot(X, Y2, label='5 iterations')
axs[0].plot(X, Y3, label='8 iterations')
axs[1].plot(X, Y4, label='8 iterations')
axs[0].plot(X, Y5, label='10 iterations')
axs[1].plot(X, Y6, label='10 iterations')
plt.xticks(X, ['Distance Difference', 'MSE(2D)', 'MSE(3D)'])
axs[1].legend(loc="upper left")
axs[0].legend(loc="upper left")
plt.xlabel("Methods")
axs[1].set_ylabel("Percentage")
axs[0].set_ylabel("Percentage")
plt.suptitle("Accuracy VS different iterations with 8:2 train/ test ratio")
plt.savefig("iter_comparison_82.png", dpi=1000)