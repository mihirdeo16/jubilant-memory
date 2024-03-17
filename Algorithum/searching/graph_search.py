#!/usr/bin/env python3
"""
Graph
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

from typing import Dict, Set, List

adjacencyList = Dict[int,List[int]]

def dfs(graph:adjacencyList)-> bool:


    return False

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
         1: [0, 2, 3, 4],
         2: [0,2],
         3: [1],
         4: [1]
         }

    dfs(graph, 0)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
