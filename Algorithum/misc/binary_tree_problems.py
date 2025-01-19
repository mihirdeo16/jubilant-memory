#!/usr/bin/env python3
"""
Binary Tree questions, using Python functions.
"""
import collections
from typing import List, Tuple, Dict, Any


class Node:
    """Tree Node """

    def __init__(self, value=None, left=None, right=None) -> None:
        self.val = value
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"{self.val}"


def dfs_binary_tree_r(root) -> list[Any]:

    if root is None:
        return []
    return [root.val] + dfs_binary_tree_r(root.left) + dfs_binary_tree_r(root.right)


def dfs_binary_tree(root) -> list[Any]:

    stack = [root]
    res = []
    while stack:
        node = stack.pop()
        res.append(node.val)

        if node.right:
            stack.append(node.right)

        if node.left:
            stack.append(node.left)
    return res


def bfs_binary_tree(root) -> list[Any]:

    queue = collections.deque([root])
    res = []
    while queue:
        local = []
        for _ in range(len(queue)):
            node = queue.popleft()
            local.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        res.append(local)
    return res

def pre_order_traversal(root):
    """
    N -> L -> R
    """
    if root is None:
        return []
    return [root.val] + pre_order_traversal(root.left) + pre_order_traversal(root.right)


def in_order_traversal(root):
    """
    L -> N -> R

    """
    if root is None:
        return []
    return in_order_traversal(root.left) + [root.val] + in_order_traversal(root.right)


def post_order_traversal(root):
    """
    L -> R ->  N

    """
    if root is None:
        return []
    return post_order_traversal(root.left) + post_order_traversal(root.right) + [root.val]


def all_tree_sum(root):


    if root is None:
        return 0

    leftTree = all_tree_sum(root.left)
    rightTree = all_tree_sum(root.right)

    return root.val + leftTree + rightTree


def max_height_tree(root):

    if root is None:
        return 0

    leftTree = max_height_tree(root.left)
    rightTree = max_height_tree(root.right)

    return 1 + max(leftTree, rightTree)


def max_width_tree(root):

    queue = collections.deque([(root, 0)])
    width_size = 0
    while queue:
        
        _, left_pos = queue[0]
        _, right_pos = queue[-1]
        width_size = max(width_size, right_pos-left_pos+1)

        for _ in range(len(queue)):
            node, pos = queue.popleft()

            if node.left:
                queue.append((node.left, 2*pos+1))

            if node.right:
                queue.append((node.right, 2*pos+2))

    return width_size



def main():
    """
    Sample Tree:
          3
        /   \
       11      4
      /  \       \
     5    2       1
    /              \
   8                14

    """
    root = Node(
        3,
        left=Node(11, left=Node(5,left=Node(8)), right=Node(2)),
        right=Node(4, right=Node(1,right=Node(14)))
    )

    # Perform DFS using recursive call stack
    res = dfs_binary_tree_r(root)
    print(f"depth first traversal recursive: {res}")

    # Perform DFS using stack
    res = dfs_binary_tree(root)
    print(f"depth first traversal using stack: {res}")

    # Perform BFS using queue
    res = bfs_binary_tree(root)
    print(f"breath first traversal: {bfs_binary_tree(root)}")

    # Traversal types:
    res = pre_order_traversal(root) # Node -> LeftTree -> RightTree
    print(f"Pre-order traversal {res}")

    res = in_order_traversal(root) # LeftTree -> Node -> RightTree
    print(f"In-order traversal {res}")

    res = post_order_traversal(root) # LeftTree -> RightTree ->  Node 
    print(f"Post-order traversal {res}")

    # Calculate sum of all nodes; using DFS
    print(f"sum of all tree nodes: {all_tree_sum(root)}")

    # Calculate max depth of tree
    print(f"maximum depth of tree: {max_height_tree(root)}")

    # Calculate max width of tree
    print(f"maximum width of tree: {max_width_tree(root)}")


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
