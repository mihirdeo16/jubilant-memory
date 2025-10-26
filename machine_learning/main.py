#!/usr/bin/env python3
"""
ML function implemented using core python and Numpy
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

import numpy as np
from tf_idf import TFIDF
from nearest_neighbors import euclidean_distance, cosine_similarity, KNearestNeighbors
from mlp import MultilayerPerceptron


def main() -> None:
    """ Main entry point of the app """
    vector_1 = np.array([3, 6, 1, 7, 4, 8])
    vector_2 = np.array([6, 1, 4, 6, 2, 6])
    cosine_val = cosine_similarity(vector_1, vector_2)
    euclidean_val = euclidean_distance(vector_1, vector_2)

    print(f"Cosine similarity: {cosine_val}")
    print(f"Euclidean distance: {euclidean_val}")

    corpus = [
        'This is the first first first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?'
    ]
    tf_idf_obj = TFIDF()
    tf_idf_obj.fit(corpus)
    # tf_idf_obj.fit_transform(corpus)
    tfidf_vec = tf_idf_obj.transform(
        ["This is the first first first document."])

    samples = np.array([
        [10, 20, 30],
        [100, 110, 140],
        [15, 20, 22],
        [120, 111, 135]
    ])
    target = np.array([0, 1, 0, 1])
    x_test = np.array([26, 27, 29])

    model = KNearestNeighbors(n_neighbors=2)
    print(model.apply(samples, target, x_test))

    mlp_model = MultilayerPerceptron(4, 5, "multiclass", 5)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
