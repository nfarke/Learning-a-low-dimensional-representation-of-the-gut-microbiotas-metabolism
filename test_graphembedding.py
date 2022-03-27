
from karateclub.dataset import GraphSetReader
from karateclub import Graph2Vec
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pdb
from sklearn.cluster import KMeans as km
from sklearn.metrics import pairwise_distances as pwd
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist
from scipy import cluster
    
read_dictionary = np.load('data2.npy',allow_pickle='TRUE').item()
read_dictionary1 = np.load('data3.npy',allow_pickle='TRUE').item()
read_dictionary2 = np.load('data4.npy',allow_pickle='TRUE').item()
read_dictionary3 = np.load('data5.npy',allow_pickle='TRUE').item()

#read_dictionary.update(read_dictionary1)
#read_dictionary.update(read_dictionary2)
#read_dictionary.update(read_dictionary3)


graphs = read_dictionary["Graph"]
name = read_dictionary["Name"]

graphs1 = read_dictionary1["Graph"]
name1 = read_dictionary1["Name"]

graphs2 = read_dictionary2["Graph"]
name2 = read_dictionary2["Name"]

graphs3 = read_dictionary3["Graph"]
name3 = read_dictionary3["Name"]


graphs = graphs + graphs1 + graphs2 + graphs3
name = name + name1 + name2 + name3

new_name = []
for k,namex in enumerate(name):
    idx = namex.find('_')
    new_name.append(namex[:idx])

model = Graph2Vec(wl_iterations = 2, dimensions = 300)
model.fit(graphs)
X = model.get_embedding()
X_embedded = TSNE(n_components=2, perplexity = 25).fit_transform(X)


#model1 = AgglomerativeClustering(n_clusters=None)
#model1 = model1.fit(X)
X = pdist(X,'Euclidean')
Z = cluster.hierarchy.linkage(X)

unique_classes = np.unique(new_name)
numeric_classes = []
for k,classx in enumerate(unique_classes):
    for kk,classy in enumerate(new_name):   
        idx = classy == classx
        numeric_classes.append(idx*(k+1))

numeric_classes_reshaped = np.reshape(numeric_classes,(len(unique_classes),len(graphs)))

classesx = []
for col in numeric_classes_reshaped.T:
    out = np.unique(col)
    classesx.append(out.max())


#get most frequent taxa
bin_freq = np.bincount(classesx)
clasname = np.unique(classesx)
sortedx  = np.argsort(bin_freq)
sortedy  = sortedx[206:] #class indexes with high frequency
sortedz  = sortedx[:205]
classesx = np.array(classesx)
plt.figure()
for k,classx in enumerate(sortedy):
    idx = classesx == classx
    plt.scatter(X_embedded[idx,0],X_embedded[idx,1], s = 50, label = unique_classes[classx])
    
for k,classx in enumerate(sortedz):
    idx = classesx == classx
    plt.scatter(X_embedded[idx,0],X_embedded[idx,1], s = 50, color = 'k', alpha = 0.05)
        
idx = unique_classes == 'Escherichia'
classi = sortedx[idx[85]*1]
plt.scatter(X_embedded[classi,0],X_embedded[classi,1], s = 50, color = 'k', label = unique_classes[idx],edgecolors = 'r')


plt.legend()
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')
plt.show()

plt.savefig('tSNE_gutbacteria.eps', format='eps')

#dist = pwd(X)
#uff = np.triu(dist)
#lindist = np.ndarray.flatten(uff)

#pca = PCA(n_components=2)
#pca.fit(X)
#X1 = pca.transform(X)

#plt.scatter(X1[:,0],X1[:,1])
#plt.show()
#plt.xlabel(pca.explained_variance_ratio_[0])
#plt.ylabel(pca.explained_variance_ratio_[1])

#print(pca.explained_variance_ratio_)