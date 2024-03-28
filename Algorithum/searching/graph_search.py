#!/usr/bin/env python3
"""
Graph
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

from typing import Dict, Set, List

adjacencyList = Dict[int,List[int]]

def dfs(graph:adjacencyList):
    """ Depth First Search
    
    Time Complexity: O(V+E) and Space Complexity: O(V)

    It takes a graph as input and returns the traversal order of the graph using Depth First Search
    Logic: We start with first node then visit its first neighbor and then its neighbor and so on. If we reach a node which has no neighbor to visit, we backtrack to the previous node and visit its other neighbor. We repeat this process until we visit all the nodes in the graph.
    
    """

    visted = set()
    traversal_order = []

    def dfs_util(node:int):
        visted.add(node)
        traversal_order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visted:
                dfs_util(neighbor)
    
    dfs_util(0)
    return traversal_order

def bfs(graph:adjacencyList):
    """ Breadth First Search
    
    Time Complexity: O(V+E) and Space Complexity: O(V)

    It takes a graph as input and returns the traversal order of the graph using Breadth First Search

    Logic: We start with the first node and visit all its neighbors.  Then we visit the neighbors of the neighbors at same level so on. We repeat this process until we visit all the nodes in the graph.
    """

    visted = set()
    traversal_order = []

    def bfs_util(node:int):
        visted.add(node)
        queue = [node]
        while queue:
            node = queue.pop(0)
            traversal_order.append(node)
            for neighbor in graph[node]:
                if neighbor not in visted:
                    visted.add(neighbor)
                    queue.append(neighbor)
    
    bfs_util(0)
    return traversal_order

def main():
    """ Main entry point of the app
    """

    # This is bi-directional graph
    graph = {
            0: [2, 1],
            2: [0, 5, 6],
            5: [2, 4],
            4: [5],
            6: [2],
            1: [0, 10],
            10: [1, 8],
            8: [10, 9],
            9: [8]
        }
    print(f"DFS: {dfs(graph)}")
    print(f"BFS: {bfs(graph)}")

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
