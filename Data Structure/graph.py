#!/usr/bin/env python3
"""
Graph implementation using adjacentList(HashMap), adjacentSet(HashSet) and adjacentMatrix(Matrix)
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

from typing import Dict, List, Tuple
from collections import defaultdict

edgeSet = List[Tuple[int, int]]
adjacencyList = Dict[int, List[int]]
adjacencyListW = Dict[int, Dict]
adjacencyMatrix = List[List[int]]


def edgeSet_to_adjacencyList(edge_set: edgeSet) -> adjacencyList:

    adjacency_list = defaultdict(list)

    for vertext_1, vertext_2 in edge_set:
        adjacency_list[vertext_1].append(vertext_2)
        adjacency_list[vertext_2].append(vertext_1)
    return adjacency_list

def edgeSet_to_adjacencyMatrix(edge_set: edgeSet) -> adjacencyMatrix:

    size = max(max(edge_set)) + 1
    matrix = [[0]*size for _ in range(size)]

    for edge_1, edge_2 in edge_set:
        matrix[edge_1][edge_2], matrix[edge_2][edge_1] = 1, 1
    return matrix


def adjacencyList_to_edgeSet(adjacency_list: adjacencyList) -> edgeSet:

    edge_set = []
    for vertext, vertext_list in adjacency_list.items():
        for vect_2 in vertext_list:
            if not any(set((vertext, vect_2)) == set(combination) for combination in edge_set):
                edge_set.append((vertext, vect_2))

    return sorted(list(edge_set))

def adjacencyList_to_adjacencyMatrix(adjacency_list: adjacencyList) -> adjacencyMatrix:

    size = len(adjacency_list.keys())
    matrix = [[0]*size for _ in adjacency_list.keys()]

    for edge, edge_list in adjacency_list.items():
        for edge_2 in edge_list:
            matrix[edge][edge_2] = 1
    return matrix


def adjacencyMatrix_to_adjacencyList(adjacency_matrix: adjacencyMatrix) -> adjacencyList:

    hashmap = defaultdict(list)

    for edge_1, edge_list in enumerate(adjacency_matrix):
        for index, edge_2 in enumerate(edge_list):
            if edge_2 == 1:
                hashmap[edge_1].append(index)
    return hashmap


def adjacencyMatrix_to_edgeSet(adjacency_matrix: adjacencyMatrix) -> edgeSet:

    hashset = []

    for edge, edge_list in enumerate(adjacency_matrix):

        for edge_2, val in enumerate(edge_list):
            if val == 1 and not any(set((edge, edge_2)) == set(com) for com in hashset):
                hashset.append((edge, edge_2))
    return hashset

def weighted_adjacencyMatrix_to_adjacencyList(adjacency_matrix:adjacencyMatrix) -> adjacencyListW:

    adjacency_list = defaultdict(dict)

    for row in range(len(adjacency_matrix)):
        for col in range(len(adjacency_matrix[0])):
            if adjacency_matrix[row][col] not in [None, float('inf')] and row!=col:
                adjacency_list[row][col] = adjacency_matrix[row][col]

    return adjacency_list

def main():
    """ Main entry point of the app 

            2 ---- 0 -- 1
             \     |    /
               \   |   /
                   3
                   |
                   4
    """



    data = [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 4)]

    # HashSet to HashMap
    adjacency_list = edgeSet_to_adjacencyList(data)

    # Hashset to Matrix
    adjacency_matrix = edgeSet_to_adjacencyMatrix(data)


    # HashMap to Hashset
    edge_set = adjacencyList_to_edgeSet(adjacency_list)

    # HashMap to Matrix
    adjacency_matrix = adjacencyList_to_adjacencyMatrix(adjacency_list)

    # Matrix to HashMap
    adjacency_list = adjacencyMatrix_to_adjacencyList(adjacency_matrix)

    # Matrix to Hashset
    edge_set = adjacencyMatrix_to_edgeSet(adjacency_matrix)

    # Weighted graph:
    weighted_graph_matrix = [
            [0, 4, 2, None, None],  # A -> B, A -> C
            [4, 0, 5, 10, None],    # B -> A, B -> C, B -> D
            [2, 5, 0, 3, 6],        # C -> A, C -> B, C -> D, C -> E
            [None, 10, 3, 0, 1],    # D -> B, D -> C, D -> E
            [None, None, 6, 1, 0]   # E -> C, E -> D
        ]
    weighted_graph_adjacency_list = weighted_adjacencyMatrix_to_adjacencyList(weighted_graph_matrix)
    print(weighted_graph_adjacency_list)



if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
