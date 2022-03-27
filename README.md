# Representation_Learning_Gut_Microbiome

In this project, we computationally selected common gut microbes based on their metabolic diversity.
First, the stoichiometric matrices were gathered from the AGORA dataset
Second, the metabolic networks were transformed into graphs.
Third, each graph was embedded into euclidean space using the representation learning algorithm graph2vec.
We then visualized their similary by using a t-SNE algorithm.
Using cosine similarity we could then pick a set of microbes which differ metabolically.

This work was performed for an ERC Consolidator Grant application.
