#!/usr/bin/env python3
"""
Binary Tree implementation in Python using Custom class and function.
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"


import array
import collections
import queue
from typing import List, Any


class Node:
    """Tree Node """

    def __init__(self, val=None, left=None, right=None) -> None:
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"{self.val}"


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


def array_representation(root) -> List[Any]:
    """
    Have to use Breath First Search
    """
    queue = collections.deque([(root,0)])
    array = []

    while queue:
        for _ in range(len(queue)):

            node, pos = queue.popleft()

            while len(array)<pos:
                array.append(None)
            array.append(node.val)

            if node.left:
                queue.append((node.left,2*pos+1))
            if node.right:
                queue.append((node.right,2*pos+2))

    return array


def breath_first_array(array_tree):

    if array_tree is None:
        return []

    queue = [0]
    res = []

    while queue:
        current_index = queue.pop(0)
        left_index = 2 * current_index + 1
        right_index = 2 * current_index + 2

        if current_index < len(array_tree):
            res.append(array_tree[current_index])

            if left_index < len(array_tree):
                queue.append(left_index)

            if right_index < len(array_tree):
                queue.append(right_index)

    return res


def depth_first_array(array_tree):

    if array_tree is None:
        return []

    stack = [0]
    res = []

    while stack:

        current_index = stack.pop()

        left_index = 2 * current_index + 1
        right_index = 2 * current_index + 2

        if current_index < len(array_tree):

            res.append(array_tree[current_index])

            if right_index < len(array_tree):
                stack.append(right_index)

            if left_index < len(array_tree):
                stack.append(left_index)

    return res


def depth_first_array_recursive(array_tree, index=0):
    if array_tree is None or index >= len(array_tree):
        return []
    if len(array_tree) < 1:
        return array_tree[index]
    return [array_tree[index]] + depth_first_array_recursive(array_tree, 2*index+1) + depth_first_array_recursive(array_tree, 2*index+2)


class BTIterator:
    def __init__(self) -> None:
        pass

    def array_representation(self, root):

        array = []
        queue = collections.deque([(root, 0)])

        while queue:
            for _ in range(len(queue)):

                node, pos = queue.popleft()

                while len(array) < pos:
                    array.append(None)
                
                if node.left:
                    queue.append((root.left,2*pos+1))

                if node.right:
                    queue.append((root.right,2*pos+2))
                array.append(node.val)

        return array

    def serialization(self, root):
        array = array_representation(root)
        array_str = [str(ele) if ele is not None else "Null" for ele in array]
        str_array = "|".join(array_str)
        return str_array

    def deserialization(self, str_array):
        array_str = str_array.split("|")
        # array = [int(ele) if ele != "Null" else None for ele in array_str]
        root = self.array_to_nodes_resp(array_str)
        return root

    def array_to_nodes_resp(self, array):

        root = Node(val=array[0])
        queue = collections.deque([(root, 0)])

        while queue:
            for _ in range(len(queue)):

                node, pos = queue.popleft()

                left_child_idx = 2*pos+1
                right_child_idx = 2*pos+2


                if left_child_idx < len(array) and array[left_child_idx]!="Null":

                    node.left = Node(val=array[left_child_idx])
                    queue.append((node.left, left_child_idx))

                if right_child_idx < len(array) and array[right_child_idx]!="Null" :

                    node.right = Node(val=array[right_child_idx])
                    queue.append((node.right, right_child_idx))

        return root
    


def main():
    """
    Sample Tree:
          3
        /   \
       11      4
      /  \       \
     5    2       1
    /            /  \
   8            7    14

    """
    root = Node(
        3,
        left=Node(11,
                  left=Node(5,
                            left=Node(8)
                            ),
                  right=Node(2)
                  ),
        right=Node(4,
                   right=Node(1,
                              left=Node(7),
                              right=Node(14)
                              )
                   )
    )

    # Perform DFS using stack
    res = dfs_binary_tree(root)
    print(f"depth first traversal using stack: {res}")

    # Perform BFS using queue
    res = bfs_binary_tree(root)
    print(f"breath first traversal: {bfs_binary_tree(root)}")

    print(f"\narray representation: {array_representation(root)}")

    # Tree serialization and deserialization using Class
    print("\n")
    binary_tree_obj = BTIterator()
    serialize_tree = binary_tree_obj.serialization(root)
    print(f"Serialize tree representation, BFS: {serialize_tree}")
    deserialize_tree = binary_tree_obj.deserialization(serialize_tree)
    res = bfs_binary_tree(deserialize_tree)
    print(f"Deserialize tree traversal using BFS: {res}")


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
