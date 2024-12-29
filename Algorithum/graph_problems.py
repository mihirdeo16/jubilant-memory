#!/usr/bin/env python3
"""
Graph problems for practice purpose, involve finding, calculating vertex & grid problems
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

from typing import Dict, List

adjacencyList = Dict[int, List[int]]


def hasPath(graph, st, dst):
    """
    Time Complexity: O(V+E)
    Space Complexity: O(V)

    It takes a graph as input, start vertex and destination vertex and return bool of True or False. 

    Logic: Iterate over all vertexes of graph, check if that is in visited vertex if not, start exploring that 
    vertex using DFS or BFS to check does start and destination has connection or not of that connected component

    """
    visited = set()

    def bfs_utils(graph, st, dst, visited):
        queue = [st]

        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                if vertex == dst:
                    return True

                visited.add(vertex)

            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    queue.append(neighbor)

        return False

    def dfs_utils(graph, st, dst, visited):
        if st == dst:
            return True
        visited.add(st)
        for neighbor in graph[st]:
            if neighbor not in visited:
                if dfs_utils(graph, neighbor, dst, visited):
                    return True
        return False

    if dfs_utils(graph, st, dst, visited):  # bfs_utils(graph,st,dst,visited)
        return True

    return False


def count_connected_components(graph: adjacencyList):
    """
    Time Complexity: O(V+E)
    Space Complexity: O(1)

    It takes a graph as input and return number of connected components, 

    Logic: Iterate over all vertexes of graphs check if exist in visited node, if not then iterate over all vertexes using DFS or BFS and count this has one component, repeat do for all vertexes.
    """

    if graph is None:
        return 0

    def dfs_utils(graph, vertex, visited):
        visited.add(vertex)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                dfs_utils(graph, neighbor, visited)

    def bfs_utils(graph, vertex, visited):
        queue = [vertex]
        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                for neighbor in graph[vertex]:
                    if neighbor not in visited:
                        queue.append(neighbor)
        return True

    visited = set()
    count = 0
    for vertex in graph:
        if vertex not in visited:

            #  ----- Using BFS ------
            # bfs_utils(graph,vertex,visited)
            # ------ Using DFS ------

            dfs_utils(graph, vertex, visited)
            count += 1

    return count


def largest_connected_components(graph):
    """
    Time Complexity: O(V+E)
    Space Complexity: O(V)

    It takes a graph as input and return largest of connected component length, 

    Logic: Iterate over all vertexes of graph, check if that is in visited vertex if not, start exploring that 
    vertex using DFS or BFS to find length of that connected component. Return that, and check if it max so far we 
    found and return that.
    """
    if graph is None:
        return 0
    visited = set()
    max_count = 0

    def dfs_utils(graph, vertex, visited):
        if vertex not in visited:
            visited.add(vertex)
            count = 1
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    count += dfs_utils(graph, neighbor, visited)

        return count

    def bfs_utils(graph, vertex):
        queue = [vertex]
        count = 0
        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                count += 1
                for neighbor in graph[vertex]:
                    if neighbor not in visited:
                        queue.append(neighbor)

        return count

    for vertex in graph:
        if vertex not in visited:

            # # ----- Using BFS ------
            # count = bfs_utils(graph,vertex)

            # ----- Using DFS ------
            count = dfs_utils(graph, vertex, visited)

            if count > max_count:
                max_count = count

    return max_count


def shortest_path(graph, start, destination):

    queue = [[start, 0]]
    visited = set()

    current_dist = -1

    while queue:

        node_tuple = queue.pop(0)

        current_node = node_tuple[0]
        current_dist = node_tuple[1]

        if current_node == destination:
            return current_dist

        current_dist += 1
        visited.add(current_node)

        for neighbor in graph[current_node]:
            if neighbor not in visited:
                queue.append([neighbor, current_dist])

    return current_dist


def count_islands(grid):
    """
    Time Complexity: O(R*C)
    Space Complexity: O(R*C)

    It takes a grid as input, return number island. 

    Logic: First convert the grid into graph by putting Bounds on rows and column, then use count connect component logic to find count. Moreover, utilize DFS or BFS to explore neighbors here, BFS could be better, and lastly create unique vertex from position of rowXcol

    """
    count = 0
    visited = set()

    def explore_island(grid, row, col, visited):

        # 1. Check if row and col coordinated going off the grid
        rowInbound, colInbound = len(grid), len(grid[0])
        if row >= rowInbound or row < 0:
            return False
        if col >= colInbound or col < 0:
            return False

        # 2. If valid, coordinated then check if we are hitting water
        if grid[row][col] == "W":
            return False

        # 3. If not water, then check is it already explored island
        pos = str(row)+", "+str(col)
        if pos in visited:
            return False

        visited.add(pos)  # Add the visited things

        explore_island(grid, row+1, col, visited)
        explore_island(grid, row-1, col, visited)
        explore_island(grid, row, col+1, visited)
        explore_island(grid, row, col-1, visited)

        return True

    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if explore_island(grid, row, col, visited):
                count += 1
    return count

    def explore(gride, row, col, visited):

        pos = str(row) + "," + str(col)

        rowInbound, colInbound = len(gride), len(gride[0])
        if ((row >= rowInbound) and (row >= 0)) or ((col >= colInbound) and (col >= 0)):
            return False

        if gride[row][col] == "W":
            return False
        if pos in visited:
            return False

        visited.add(pos)

        explore(gride, row+1, col, visited)
        explore(gride, row-1, col, visited)
        explore(gride, row, col-1, visited)
        explore(gride, row, col+1, visited)

        return True

    visited = set()
    count = 0
    for row in range(len(gride)):
        for col in range(len(gride[0])):
            if explore(gride, row, col, visited):
                count += 1
    return count


def find_maximum_size_island(grid):
    """
    Time Complexity: O(R*C)
    Space Complexity: O(R*C)

    It takes a grid as input, return number island with maximum size. 

    Logic: First convert the grid into graph by putting bounds on rows and column, then use count connect component logic to find island and count the elements init. Moreover, utilize DFS or BFS to explore neighbors here, and lastly create unique vertex from position of rowXcol

    """
    if grid is None:
        return 0

    visited = set()
    max_count = 0

    def explore_length_island(grid, row, col, visited):

        # 1. Things go out of bound
        rowBound, colBound = len(grid), len(grid[0])
        if row >= rowBound or row < 0:
            return 0
        if col >= colBound or col < 0:
            return 0

        # 2. Hit the water body
        if grid[row][col] == "W":
            return 0

        # 3. If it belong to already visited land
        pos = str(row)+", "+str(col)
        if pos in visited:
            return 0

        visited.add(pos)
        count = 1
        count += explore_length_island(grid, row+1, col, visited)
        count += explore_length_island(grid, row-1, col, visited)
        count += explore_length_island(grid, row, col+1, visited)
        count += explore_length_island(grid, row, col-1, visited)

        return count

    for row in range(len(grid)):
        for col in range(len(grid)):
            count = explore_length_island(grid, row, col, visited)
            max_count = max(count, max_count)
    return max_count


def find_minimum_size_island(grid):
    """
    Time Complexity: O(R*C)
    Space Complexity: O(R*C)

    It takes a grid as input, return number island with minimum size. 

    Logic: First convert the grid into graph by putting bounds on rows and column, then use count connect component logic to find island and count the elements init. Moreover, utilize DFS or BFS to explore neighbors here, and lastly create unique vertex from position of rowXcol

    """
    if grid is None:
        return 0

    visited = set()
    min_island = float('inf')

    def explore_length_island(grid, row, col, visited):

        # 1. Things go out of bound
        rowBound, colBound = len(grid), len(grid[0])
        if row >= rowBound or row < 0:
            return 0
        if col >= colBound or col < 0:
            return 0

        # 2. Hit the water body
        if grid[row][col] == "W":
            return 0

        # 3. If it belong to already visited land
        pos = str(row)+", "+str(col)
        if pos in visited:
            return 0

        visited.add(pos)
        count = 1
        count += explore_length_island(grid, row+1, col, visited)
        count += explore_length_island(grid, row-1, col, visited)
        count += explore_length_island(grid, row, col+1, visited)
        count += explore_length_island(grid, row, col-1, visited)

        return count

    for row in range(len(grid)):
        for col in range(len(grid[0])):
            count = explore_length_island(grid, row, col, visited)
            if count > 0:
                min_island = min(count, min_island)
    return min_island


def main():
    """ Main entry point of function.
                   0               11 --- 12        14 --- 15      16
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

    print(
        f"Count of connected components : {count_connected_components(graph)}")

    print(
        f"Length of largest connected components: {largest_connected_components(graph)}")

    ele_1, ele_2 = 4, 16
    print(f"Has path between {ele_1} to {ele_2}: {hasPath(graph,ele_1,ele_2)}")

    print(
        f"Shortest path between {ele_1} to {ele_2}: {shortest_path(graph,ele_1,ele_2)}")

    grid = [
        ['W', 'L', 'W', 'W', 'L'],
        ['W', 'L', 'W', 'W', 'W'],
        ['W', 'W', 'W', 'L', 'W'],
        ['W', 'W', 'L', 'L', 'W'],
        ['L', 'W', 'W', 'L', 'L'],
        ['L', 'L', 'W', 'L', 'W'],
    ]

    print(f"\n\nCount total island : {count_islands(grid)}")
    print(f"Find island with maximum size: {find_maximum_size_island(grid)}")
    print(f"Find island with minimum size: {find_minimum_size_island(grid)}")


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
