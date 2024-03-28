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

    visted = set()
    traversal_order = []

    def dfs_util(node:int):
        visted.add(node)
        traversal_order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visted:
                dfs_util(neighbor)
    
    dfs_util(0)
    print(traversal_order)

def bfs(graph:adjacencyList):

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
    print(traversal_order)

def main():
    """ Main entry point of the app
          4 
          |
    0 ----1-----3
    |     |
    |-----2
    
    """
    graph = {
            0: [1, 2],
            1: [2, 4],
            2: [0 ],
            3: [],
            4: [3],
        }
    print(f"Graph: {graph}")
    print(f"DFS: ")
    dfs(graph)
    print(f"BFS: ")
    bfs(graph) 
if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
