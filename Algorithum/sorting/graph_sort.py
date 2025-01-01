#!/usr/bin/env python3
"""
Topological Sort:
+ Topological ordering is an ordering of the node where find directed node from root to child.
+ It only work with acyclic directed graph, (DAG).
+ To check if it is DAG: Tarjan's strongly connected component algorithm
+ Topological ordering are not unique. 

Time   Complexity: O(V+E)
Memory Complexity: O(V)

Examples:
1. Find valid order of installing dependency in python setup.
2. Can all class be taken with prerequisite
"""

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


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
