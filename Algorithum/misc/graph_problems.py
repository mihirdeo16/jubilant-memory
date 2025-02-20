#!/usr/bin/env python3
"""
Graph problems for practice purpose, involve finding, calculating vertex & grid problems
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

from typing import Dict, List
import collections
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
    max_count = 0

    if graph is None:
        return max_count
    
    visited = set()
    def dfs_utils(graph, vertex, visited):
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

def loop_detection(graph):
    visited = set()
    start_vertex = next(iter(graph))

    recursion_stack = set()

    def dfs_utls(graph, start_vertex):

        visited.add(start_vertex)

        recursion_stack.add(start_vertex)

        for neighbor in graph[start_vertex]:
            if neighbor not in visited:
                if dfs_utls(graph, neighbor):
                    return True
            elif neighbor in recursion_stack:
                return True

        recursion_stack.remove(start_vertex)
        return False

    for vertex in graph:
        if vertex not in visited:
            if dfs_utls(graph, start_vertex):
                return True
    return False

def findOrder(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    To be course called as valid where they are DAG, we can use topological sort, convert the edgeSet to adjacencyList. then apply sort
    """
    # Create graph of adjacencyList fro edgeSet
    graph = collections.defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)

    ordering = []
    visited = set()
    loop_check = set()

    def topological_sort(node,graph,visited,loop_check):
        if node in loop_check:
            return False
        
        loop_check.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                if not topological_sort(neighbor,graph,visited,loop_check):
                    return False
        
        visited.add(node)
        ordering.append(node)
        loop_check.remove(node)
        return True


    for course in range(numCourses): 
        if course not in visited:
            if not topological_sort(course,graph,visited,loop_check): # Loop is detected
                return []
    
    return ordering[::-1]
            
        


def cleanRoom(robot):

    visited = set()
    directions = [(0,1),(1,0),(0,-1),(-1,0)] # UP, RIGHT, DOWN, LEFT


    def backtrack(x,y,curr_direction):
        visited.add((x,y))
        robot.clean()

        for i in range(len(directions)):

            new_direction = (curr_direction + i) % 4
            new_x, new_y = x + directions[new_direction][0], y + directions[new_direction][1]

            if (new_x, new_y) not in visited and robot.move():
                backtrack(new_x,new_y,new_direction)
                
                # Reset the same spot
                robot.turnLeft()
                robot.turnLeft()
                robot.move()
                robot.turnLeft()
                robot.turnLeft()
        robot.turnLeft()

    backtrack(x=0,y=0,curr_direction=0)


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


def longestIncreasingPath(matrix: List[List[int]]) -> int:
    visited = {}

    def dfs_find_path(row,col,matrix,visited,prevValue):
        rowBound, colBound = len(matrix), len(matrix[0])
        if (row <0 or row >= rowBound) or (col <0 or col >= colBound):
            return 0
        
        if matrix[row][col] <= prevValue:
            return 0
        
        if (row,col) in visited:
            return visited[(row,col)]
        
        res = 1

        res = max(res, 1+dfs_find_path(row+1,col,matrix,visited,matrix[row][col]) )
        
        res = max(res, 1+ dfs_find_path(row-1,col,matrix,visited,matrix[row][col]) )
        res = max(res, 1+ dfs_find_path(row,col-1,matrix,visited,matrix[row][col]) )
        res = max(res, 1 + dfs_find_path(row,col+1,matrix,visited,matrix[row][col]) )

        return res
        
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            dfs_find_path(row,col,matrix,visited,-1)

    return max(visited.values())

def unique_path(obstacleGrid):
    if obstacleGrid[-1][-1]==1 or obstacleGrid[0][0]==1 :
        return 0
    if len(obstacleGrid)==1:
        return 1 if 1 not in obstacleGrid[0] else 0
    
    memo = {}
    def explore_path(x,y,obstacleGrid,memo):
        rowBound, colBound = len(obstacleGrid), len(obstacleGrid[0])
        # Out of bound issue
        if x >= rowBound or y >= colBound:
            return 0
        
        # We met the Obstacle
        if obstacleGrid[x][y]==1:
            return 0
        
        # We met the destination
        if x == rowBound -1 and y == colBound -1:
            return 1
        
        # Already visited
        if (x,y) in memo:
            return memo[(x,y)]
        
        # Explore 
        right_dir_path = explore_path(x+1,y,obstacleGrid,memo)
        down_dir_path = explore_path(x,y+1,obstacleGrid,memo)

        memo[(x,y)] = right_dir_path + down_dir_path

        return memo[(x,y)]
    
    explore_path(0,0,obstacleGrid,memo)

    return memo[(0,0)]

def calEquation(equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
    """
    T - O(M * N), M- No. of queries, N - No. of equations
    M - O(N) - No of equations
    
    """

    # Create graph of adjacency list of weighted graph
    graph = collections.defaultdict(dict)

    for equa_var, val in zip(equations,values):
        graph[equa_var[0]][equa_var[1]]= val
        graph[equa_var[1]][equa_var[0]] = 1 /val 

    def backtrack(src,dst,graph,acc,visited):
        # If src is same as dst, return acc
        if src ==dst:
            return acc
        
        # Add node to visited set
        visited.add(src)

        # Explore
        for neighbor, val in graph[src].items():
            if neighbor not in visited:
                result = backtrack(neighbor,dst,graph,acc*val,visited)
                if result != -1.0:
                    return result
        return -1.0


    res = []
    for query in queries:
        # Check if query[0] to query[1] is valid variables
        src, dst = query[0], query[1]
        if src not in graph or dst not in graph:
            res.append(-1.0)
        else:
            # If they are valid, now calculate
            visited = set()
            acc = 1
            acc = backtrack(src,dst,graph,acc,visited)
            res.append(acc)
    return res
            
        


def find_good_farmland(grid:List[List[int]])-> float:
    """
    So in technical terms, we can call this as graph search problem, where we need to find out max square we can form.
    We can slove this using backtracking, recursive call. Where we first define base conditions as backtrack on it.
    We can treat this grid as coordinate space, adjacency matrix.
    1) If we found 1, then its max_area = 1
    2) To expand its right, down and diagonal should be 1.
    3) Also we need to take care of boundaries.
    4) To reduce husle of revisiting corrodinates we can use memo
    
    let say we have r - rows and c -cols then
    T - O(r*c)
    M - O(r*c)
    """
    def find_area(row,col,grid,visited):

        rowBound, colBound = len(grid), len(grid[0])
        if row >= rowBound or col >= colBound:
            return 0

        if grid[row][col]==0:
            return 0

        if (row,col) in visited:
            return visited[(row,col)]

        # Explore 
        right_side = find_area(row+1,col,grid,visited) # Explore its right
        down_side = find_area(row,col+1,grid,visited) # Explore its down
        diagonal_side = find_area(row+1,col+1,grid,visited) # Explore its diagonal

        visited[(row,col)] = min(right_side,diagonal_side,down_side) + 1

        return visited[(row,col)]

    visited = {}
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            find_area(row,col,grid,visited)
    
    return max(visited.values())


def removeStones(stones: List[List[int]]) -> int:
    
    # Build graph
    graph = collections.defaultdict(list)
    for i in range(len(stones)):
        for j in range(i+1, len(stones)):
            if stones[i][0]==stones[j][0] or stones[i][1]==stones[j][1]:
                graph[i].append(j)
                graph[j].append(i)
    
    # Find connected components
    def dfs(node,graph,visited):
        visited.add(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor,graph,visited)
            
    count = 0
    visited = set()
    for i in range(len(stones)):
        if i not in visited:
            dfs(i,graph,visited)
            count += 1

    return len(graph) - count



def main():
    """ Main entry point of function.
                   0               11 --- 12        14 --- 15      16
            /              \               |
            2               1              13
         /                  \
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

    print(f"Count of connected components:{count_connected_components(graph)}")

    print(
        f"Length of largest connected components:{largest_connected_components(graph)}")

    ele_1, ele_2 = 4, 16
    print(f"Has path between {ele_1} to {ele_2}:{hasPath(graph,ele_1,ele_2)}")

    print(
        f"Shortest path between {ele_1} to {ele_2}: {shortest_path(graph,ele_1,ele_2)}")

    """
    A ---> B ---> C ---> D ---> E
    """
    graph_no_loop = {
        "A": ["B", "C"],
        "B": ["C"],
        "C": ["D"],
        "D": ["E"],
        "E": []  # No outgoing edges from E
    }

    print("Loop detection in graph_no_loop", loop_detection(graph_no_loop))
    """
    A ---> B ---> C ---> D ---> E
                  ^             |
                  |-------------|
    """
    graph_with_loop = {
        "A": ["B", "C"],
        "B": ["C"],
        "C": ["D"],
        "D": ["E"],
        "E": ["B"]  # E points back to B, creating a loop
    }
    print("Loop detection in graph_with_loop", loop_detection(graph_with_loop))

    # Grid traversal problem:
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

    grid = [
            [0,1,1,0,1],
            [1,1,0,1,0],
            [0,1,1,1,0],
            [1,1,1,1,0],
            [1,1,1,1,1],
            [0,0,0,0,0]
        ]
    print(f"Find maximum area of farming (square), {find_good_farmland(grid)}")

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
