#!/usr/bin/env python3
"""
NearestNeighbors implementation
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

import numpy as np

class KNearestNeighbors:
    """
    It is supervise machine learning algorithm, using distance to calculate likehood of new sample belong to which category. For given x_train and y_train, find for new point x_test which group of y_train it belong based on K nearest neighbors. 
    """

    def __init__(
        self, n_neighbors: int = 5, distance_type: str = "euclidean_distance"
    ) -> None:
        self.embedding = dict()
        self.n_neighbors = n_neighbors
        self.fn_mapping = {
            "euclidean_distance": self.euclidean_distance,
            "manhattan_distance": self.manhattan_distance,
            "cosine_distance": self.cosine_distance,
        }
        self.distance_fn = self.fn_mapping[distance_type]

    def euclidean_distance(self, a, b) -> np.float64:
        return np.sqrt(np.sum(np.square(b - a)))

    def manhattan_distance(self, a, b) -> np.float64:
        return np.sum(np.abs(a - b))

    def cosine_distance(self, a, b) -> np.float64:
        return 1 - (
            np.dot(a, b)
            / (np.sqrt(np.sum(np.square(a))) * np.sqrt(np.sum(np.square(a))))
        )

    def apply(
        self, samples: np.ndarray, target: np.ndarray, x_test: np.float64
    ) -> np.ndarray:

        # Calculate for x_test all distance with x_train/samples
        distance = [self.distance_fn(x, x_test) for x in samples]

        # Sort by distance and return its indices
        sorted_indices = np.argsort(distance)

        # Clip indices to n_neighbors
        k_neighbors = target[sorted_indices[: self.n_neighbors]]

        # Get majority class
        unique, counts = np.unique(k_neighbors, return_counts=True)
        k_neighbors_cls = unique[np.argmax(counts)]

        return k_neighbors_cls
