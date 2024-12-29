#!/usr/bin/env python3
"""
Graph
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

from typing import Dict, Set, List

adjacencyList = Dict[int, List[int]]


def dfs(graph: adjacencyList)  :
    """ 
    Depth First Search

    Time Complexity: O(V+E)
    Space Complexity: O(V)

    It takes a graph as input and returns the traversal order of the graph using Depth First Search.

    Logic: It uses stack to store the order, and set to keep track. Strategy here is to exploit one vertex and going as deep as possible 
    along each branch before backtracking. We start with first node in graph, add to stack.While our stack is not empty, we pop last element, 
    perform desire operation on it, (add to list, check if target etc.)Then add as visited in set, so we don't visit again. 
    Then iterate over all connected elements, add to our stack. This conclude our DFS with stack.

    """
    if graph is None:
        return ""
    
    visited = set()
    vertex = next(iter(graph))
    stack = [vertex]

    result = []
    while stack:
        vertex = stack.pop()
        visited.add(vertex)

        result.append(str(vertex))  # Save the result

        for neighbor in graph[vertex]:
            if neighbor not in visited:
                stack.append(neighbor)

    return ", ".join(result)


def dfs_recursive(graph: adjacencyList):
    """ Depth First Search recursively

    Time Complexity: O(V+E)
    Space Complexity: O(V)

    It takes a graph as input and returns the traversal order of the graph using Depth First Search

    Logic: For given graph, we first define empty set to save visited elements, then define subfunction to iterate recursively over start vertex. 
    As base condition we check does vertex already exist in our set, if not then we will perform first operation of interest (add to list, check condition) then add that element in visited set. Next iterate over all neighbors of vertex. For each vertex first we check vertex not in visited set and them we will call same subfunction with new neighbor as vertex. Here by recursive nature, we will go first as deep as possible on first 
    neighbor itself and achieve depth first logic and then others. 
    Lastly, some times you may have not connected component and base case of empty graph need to take care of 

    """
    if graph is None:
        return ""
    
    visited = set()
    vertex = next(iter(graph))
    result = []

    def dfs_utils(vertex):
        if vertex not in visited:
            result.append(str(vertex))  # Save the result
            visited.add(vertex)
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    dfs_utils(neighbor)

    dfs_utils(vertex)

    # Loop through all nodes in the graph to handle disconnected components
    for vertex in graph:
        if vertex not in visited:
            dfs_utils(vertex)

    return ", ".join(result)


def bfs(graph: adjacencyList):
    """ Breadth First Search

    Time Complexity: O(V+E)
    Space Complexity: O(V)

    It takes a graph as input and returns the traversal order of the graph using Breadth First Search

    Logic: It uses queue to store the order, and set to keep track. Strategy here is to exploit all neighbors of given vertex before moving to step deep. 
    We start with first vertex in graph, add to queue. While our queue is not empty, we dequeue (pop) first element, and perform desire operation on it, 
    (add to list, check if target etc.). Then add as visited in set, so we don't visit again. Then iterate over all connected neighbors, add to our queue. 
    This conclude our BFS with queue.
    """

    if graph is None:
        return ""
    visited = set()
    vertex = next(iter(graph))
    queue = [vertex]

    result = []

    while queue:

        vertex = queue.pop(0)
        visited.add(vertex)

        result.append(str(vertex))  # Save result

        for neighbor in graph[vertex]:
            if neighbor not in visited:
                queue.append(neighbor)

    return ", ".join(result)


def main():
    """ Main entry point of the app
                   0               11 --- 12        14 --- 15
            /              \               |
            2               1              13
         /    \              \
        5 ---- 6             10
      /                          \
    4                               8
                                      \
                                        9

    """

    # This is bi-directional graph
    graph = {
        0: [2, 1],
        1: [0, 10],
        2: [0, 5, 6],
        5: [2, 4, 6],
        4: [5],
        6: [2],
        10: [1, 8],
        8: [10, 9],
        9: [8],
        11: [12],
        12: [11, 13],
        13: [12],
        14: [15],
        15: [14],
        16: []
    }

    print(f"Depth First Search: {dfs(graph)}")

    print(f"Depth First Search recursive: {dfs_recursive(graph)}")

    print(f"Breath First Search: {bfs(graph)}")

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
