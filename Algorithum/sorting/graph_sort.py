#!/usr/bin/env python3
"""

Graph algorithms:


----------- Topological Sort:
+ Topological ordering is an ordering of the node where find directed node from root to child.
+ It only work with acyclic directed graph, (DAG).
+ To check if it is DAG: Tarjan's strongly connected component algorithm
+ Topological ordering are not unique. 
    Time   Complexity: O(V+E)
    Memory Complexity: O(V)
Examples:
    1. Find valid order of installing dependency in python setup.
    2. Can all class be taken with prerequisite

----------- Dijkstra's Shortest Path Algorithm:
+ It work with with weighted graph.
+ Problem is given node we want to find out shortest distance of every node from it.
+ Logic: It is implemented using Priority queue and update the weights.
+ Topological ordering are not unique. 
    Time   Complexity: O(V+E) * O(v)
    Memory Complexity: O(V+E)

"""

import heapq
__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"


def reverse_list(t_list):

    if len(t_list) <= 0:
        return []

    return reverse_list(t_list[1:]) + [t_list[0]]


def topological_sort(graph):
    ordering = []
    visited = set()
    start = next(iter(graph))
    check_cycle = set()

    def dfs_utils(graph, start, visited, check_cycle):

        if start in check_cycle:
            raise ValueError("Cycle found.")

        visited.add(start)

        check_cycle.add(start)

        for neighbor in graph[start]:
            if neighbor not in visited:
                dfs_utils(graph, neighbor, visited, check_cycle)

        check_cycle.remove(start)

        ordering.append(start)

    for vertex in graph:
        if vertex not in ordering:
            dfs_utils(graph, start, visited, check_cycle)

    return reverse_list(ordering)


def dijkstra_shortest_path(graph):
    """
    Dijkstra's Shortest Path Algorithm
    
    """
    visited = set()
    start_node = next(iter(graph)) # This point form which we will calculate
    distance_table = {node:float('inf') for node in graph}

    distance_table[start_node] = 0
    queue = [(0,start_node)] 

    heapq.heapify(queue)

    while queue:

        curr_dist, curr_node = heapq.heappop(queue)

        if curr_node not in visited:
            visited.add(curr_node)

            for neighbor, distance in graph[curr_node].items():
                if neighbor not in visited:
                    new_distance = distance + curr_dist
                    if new_distance < distance_table[neighbor]:
                        distance_table[neighbor] = new_distance
                        heapq.heappush(queue,(new_distance,neighbor))

    return distance_table

def main():
    """ Main entry point of the app """
    graph = {
        'A': ['B', 'C'],
        'B': ['D'],
        'C': ['E','F'],
        'D':[],
        'E': [],
        'F':[]
    }
    stack = topological_sort(graph)

    print(stack)

    # Example usage:
    graph = {
        'A': {'B': 1, 'C': 4},
        'B': {'A': 1, 'C': 2, 'D': 5},
        'C': {'A': 4, 'B': 2, 'D': 1},
        'D': {'B': 5, 'C': 1}
    }

    shortest_paths = dijkstra_shortest_path(graph)
    print(shortest_paths)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
