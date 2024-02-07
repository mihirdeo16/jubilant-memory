#!/usr/bin/env python3
"""
Graph 
"""

from typing import Dict
from collections import defaultdict
__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

"""
    Searching algo:
        DFS - O(n)
            HashSet
        BFS - O(n)
            Queue Double ended 
            Hashset
    Union O(n logn)
        Forest of Tree
        To find connected component 
    Topological Sort O(n)
        Hashset
    Dijkstra's Shortest Path Algo (E log(V))
        Heap, HashSet

"""


def convert_adjacency_list(edge_set) -> Dict[int,list]:

    adjacency_list = defaultdict(list)

    for vertext_1, vertext_2 in edge_set:
        adjacency_list[vertext_1].append(vertext_2)
        adjacency_list[vertext_2].append(vertext_1)
    return adjacency_list


def convert_edge_set(adjacency_list) -> list[tuple]:

    edge_set = set()
    for vertext, vertext_list in adjacency_list.items():
        for vect_2 in vertext_list:
            if any(set((vertext, vect_2)) == set(combination) for combination in edge_set):
                pass
            else:
                edge_set.add((vertext, vect_2))

    return sorted(list(edge_set))


def convert_adjacency_matrix(edge_set) -> list[list[int]]:

    size = max(max(edge_set))
    matrix = [[0]*size for _ in range(size)]

    for edge_1, edge_2 in edge_set:
        matrix[edge_1][edge_2], matrix[edge_2][edge_1] = 1, 1
    return matrix


def main():
    """ Main entry point of the app """

    data = {(0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 4)}

    data_adjacency_list = convert_adjacency_list(data)  # HashMap
    data_edge_set = convert_edge_set(data_adjacency_list)  # Hashset
    data_adjacency_matrix = convert_adjacency_matrix(data)  # List of list


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
