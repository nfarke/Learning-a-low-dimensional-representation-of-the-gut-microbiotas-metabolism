# -*- coding: utf-8 -*-

import scipy.io
from os import listdir
import numpy as np
from scipy import sparse
import pdb
import networkx as nx
import matplotlib.pyplot as plt

dirx = 'C:/Users/Niklas/Documents/PhD/microbiome/AGORA/CurrentVersion/AGORA_1_03/AGORA_1_03_mat/'
listx = listdir(dirx)
listx1 = listx[625:]
mydic = {}
Graphs = {"Name":[],"Graph":[]};
for k in range(683,818):
    print(k)
    filex = listx[k]
    stringx = dirx+filex
    matx = scipy.io.loadmat(dirx+filex)
    
    S = matx['model']['S']
    Sd = S[0][0].todense()
    Sd1 = Sd != 0
    Sd_bool = Sd1*1
    Sd_adj = Sd_bool*np.transpose(Sd_bool)
    Sd_adj = Sd_adj != 0
    Sd_adj = Sd_adj*1
    
    G=nx.from_numpy_matrix(Sd_adj)
    
    # Generate connected components and select the largest:
    largest_component = max(nx.connected_components(G), key=len)

    # Create a subgraph of G consisting only of this component:
    G2 = G.subgraph(largest_component)
    new_nodes = np.arange(len(G2.nodes))
    
    i = 0    
    for node in G2.nodes:
        mydic[node] = new_nodes[i]
        i = i + 1
    
    G2 = nx.relabel.relabel_nodes(G2,mydic)
        
    Graphs["Name"].append(filex)
    Graphs["Graph"].append(G2)
    np.save('data5.npy', Graphs)    
  #  pos = nx.spring_layout(G2)
  #  nx.draw_networkx_nodes(G2, pos, node_size = 100)
   # nx.draw_networkx_labels(G2, pos)
   # nx.draw_networkx_edges(G2, pos)
   # nx.draw_networkx_edges(G2, pos)
   # plt.show()
    #disp(Sd.shape)




#read_dictionary = np.load('my_file.npy',allow_pickle='TRUE').item()