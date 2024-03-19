#!/usr/bin/env python3
"""
NearestNeighbors implementation
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

import numpy as np

def cosine_similarity(a,b) -> np.float64:
    return np.dot(a,b)/ (np.sqrt(np.sum(np.square(a)))*np.sqrt(np.sum(np.square(b))))

def euclidean_distance(a,b) -> np.float64:
    return np.sqrt(np.sum(np.square(b-a)))

def manhattan_distance(a,b) -> np.float64:
    return np.sum(np.abs(a - b))

def cosine_distance(a,b) -> np.float64:
    return 1 - (np.dot(a,b)/(np.sqrt(np.sum(np.square(a)))* np.sqrt(np.sum(np.square(a)))))
class KNearestNeighbors:

    def __init__(self,n_neighbors:int=5) -> None:
        self.embedding = dict()
        self.n_neighbors = n_neighbors

    def apply(self,samples,target,x_test) -> None:

        distance = [ euclidean_distance(x,x_test) for x in samples]
        sorted_indices = np.argsort(distance)

        k_neighbors = target[sorted_indices[:self.n_neighbors]]

        # Get majority class
        unique, counts = np.unique(k_neighbors, return_counts=True)
        k_neighbors_cls = unique[np.argmax(counts)]

        return k_neighbors_cls