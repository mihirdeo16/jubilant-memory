#!/usr/bin/env python3
"""
Binary Tree implementation in Python using Custom class and function.
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"


from typing import List, Any 
import collections

class Node:
    """Tree Node """

    def __init__(self, value=None, left=None, right=None) -> None:
        self.value = value
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"{self.value}"


def depth_first_traversal_recursive(root):

    if root is None:
        return []

    return [root.value] + depth_first_traversal_recursive(root.left) + depth_first_traversal_recursive(root.right)


def depth_first_traversal(root):

    if root is None:
        return []

    traversal_stack = [root]
    res = []
    while traversal_stack:

        node = traversal_stack.pop()
        res.append(node.value)

        if node.right:
            traversal_stack.append(node.right)
        if node.left:
            traversal_stack.append(node.left)

    return res

def breath_first_traversal(root):

    if root is None:
        return []

    traversal_queue = collections.deque([root])
    res = []
    while traversal_queue:

        for _ in range(len(traversal_queue)):
            node = traversal_queue.popleft()
            res.append(node.value)

            if node.left:
                traversal_queue.append(node.left)
            if node.right:
                traversal_queue.append(node.right)

    return res


def all_tree_sum(root):

    if root is None:
        return 0

    return root.value + all_tree_sum(root.left) + all_tree_sum(root.right)

def diameter_max_height_tree(root):

    res = [0]
    def height_tree(root):
        if root is None:
            return 0
        
        leftTree = max_height_tree(root.left)
        rightTree = max_height_tree(root.right)

        res[0] = max(res[0],1 + leftTree + rightTree )

        return 1 + max(leftTree,rightTree)
    height_tree(root)
    return res[0]

def max_height_tree(root):
    if root is None:
        return 0
    
    leftTree = max_height_tree(root.left)
    rightTree = max_height_tree(root.right)

    return 1 + max(leftTree,rightTree)

def max_width_tree(root):
    width = 0
    queue = collections.deque([(root,0)])
    while queue:
        _, first_pos = queue[0]
        _, last_pos = queue[-1]

        for _ in range(len(queue)):
            node, pos = queue.popleft()

            if node.left:
                queue.append((node.left,2*pos+1))
            if node.right:
                queue.append((node.left,2*pos+2))
        width = max(width,last_pos-first_pos)
    return width

def pre_order_traversal(root):
    """
    N -> L -> R
    """
    if root is None:
        return []]
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
    return post_order_traversal(root.left) + post_order_traversal(root.right)+ [root.val] 

def array_representation(root) -> List[Any]:
    """
    Have to use Breath First Search
    """
    queue = [root]
    array= []

    while queue:

        node = queue.pop(0)
        if node:

            array.append(node.value)

            queue.append(node.left)
            queue.append(node.right)

        else:
            array.append(None)
    
    while array and array[-1] is None:
        array.pop()

    return array

def breath_first_array(array_tree):

    if array_tree is None:
        return []
    
    queue= [0]
    res = []

    while queue:
        current_index = queue.pop(0)
        left_index = 2* current_index + 1
        right_index = 2* current_index + 2


        if current_index < len(array_tree) :
            res.append(array_tree[current_index])

            if left_index < len(array_tree):
                queue.append(left_index)

            if right_index < len(array_tree):
                queue.append(right_index)
            
    return res

def depth_first_array(array_tree):

    if array_tree is None:
        return []
    
    stack= [0]
    res = []

    while stack:

        current_index = stack.pop()

        left_index = 2* current_index + 1
        right_index = 2* current_index + 2

        if current_index < len(array_tree) :
        
            res.append(array_tree[current_index])

            if right_index < len(array_tree):
                stack.append(right_index)

            if left_index < len(array_tree):
                stack.append(left_index)

    return res

def depth_first_array_recursive(array_tree,index=0):
    if array_tree is None or index >= len(array_tree):
        return []
    if len(array_tree) < 1:
        return array_tree[index]
    return [array_tree[index]] + depth_first_array_recursive(array_tree,2*index+1) + depth_first_array_recursive(array_tree,2*index+2)


def main():
    """
    Sample Tree:
          3
        /   \
       11      4
      /  \       \
     5    2       1

    """
    root = Node(
                3, 
                left=Node(11, left=Node(4), right=Node(2)), 
                right=Node(5, right=Node(1))
                )

    print(f"Depth first traversal: {depth_first_traversal(root)}")
    print(f"Depth first traversal recursively: {depth_first_traversal_recursive(root)}")

    print(f"Breath first traversal: {breath_first_traversal(root)}")

    print(f"sum of all tree nodes: {all_tree_sum(root)}")
    # print(f"tree_min_value: {tree_min_value(root)}")
    print(f"maximum depth of tree: {max_height_tree(root)}")


    print(f"\narray_representation: {array_representation(root)}")
    array_tree = array_representation(root)

    print(f"breath_first_array: {breath_first_array(array_tree)}")


    print(f"depth_first_array: {depth_first_array(array_tree)}")
    print(f"depth_first_array_recursive: {depth_first_array_recursive(array_tree)}")



if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
