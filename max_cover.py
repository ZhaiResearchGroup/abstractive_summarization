import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys

'''
adj_matrix: A matrix where each sentence is adjacent by some weight
clusters  : Number of sentence clusters to generate
n_per     : Number of sentences per cluster, clusters * n_per <= # sentences
returns   : numpy array where each row contains sentence indices for a cluster
'''
def max_cover(adj_matrix, topics, n_per):
    sentence_indices = np.zeros((topics, n_per))

    # perform max cover
    for topic in range(topics):
        node_covers = np.sum(adj_matrix, axis=0)
        topic_index = np.argmax(node_covers)

        # n_per closest sentences to chosen
        n_closest = np.argpartition(adj_matrix[:, topic_index], -1*n_per)[-1*n_per:]
        sentence_indices[topic, :] = n_closest

        # remove edges
        for s_index in n_closest:
            adj_matrix[s_index, :] = 0
            adj_matrix[s_index, s_index] = -np.inf 

    return sentence_indices
