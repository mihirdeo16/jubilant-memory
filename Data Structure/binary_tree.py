#!/usr/bin/env python3
"""
Binary Tree implementation in Python using Custom class and function.
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"


from re import A
from typing import List, Any 

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

    traversal_stack = [root]
    res = []
    while len(traversal_stack) > 0:

        node = traversal_stack.pop(0)
        res.append(node.value)

        if node.left:
            traversal_stack.append(node.left)
        if node.right:
            traversal_stack.append(node.right)

    return res


def breath_first_search(root, ele):

    if root is None:
        return []

    traversal_stack = [root]
    while len(traversal_stack) > 0:

        node = traversal_stack.pop(0)
        if node.value == ele:
            return True
        if node.left:
            traversal_stack.append(node.left)
        if node.right:
            traversal_stack.append(node.right)

    return False


def depth_first_search_recursive(root, ele):

    if root is None:
        return False
    if root.value == ele:
        return True

    return depth_first_search_recursive(root.left, ele) or depth_first_search_recursive(root.right, ele)


def depth_first_search(root, ele):

    if root is None:
        return []

    traversal_stack = [root]
    while traversal_stack:

        node = traversal_stack.pop()
        if node.value == ele:
            return True

        if node.right:
            traversal_stack.append(node.right)
        if node.left:
            traversal_stack.append(node.left)

    return False


def tree_sum(root):

    if root is None:
        return 0

    return root.value + tree_sum(root.left) + tree_sum(root.right)


def tree_min_value(root):

    if root is None:
        return float('inf')

    return min(root.value, tree_min_value(root.left), tree_min_value(root.right))


def max_depth_of_tree(root):
    if root is None:
        return 0
    return root.value + max(max_depth_of_tree(root.left), max_depth_of_tree(root.right))


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
     4    2       1

    """
    root = Node(
                3, 
                left=Node(11, left=Node(4), right=Node(2)), 
                right=Node(4, right=Node(1))
                )

    print(f"depth_first_traversal: {depth_first_traversal(root)}")
    print(
        f"depth_first_traversal_recursive: {depth_first_traversal_recursive(root)}")
    
    print(f"breath_first_traversal: {breath_first_traversal(root)}")


    ele = 1
    print(
        f"breath_first_search for ele {ele} : {breath_first_search(root,ele)}")

    ele = 6
    print(
        f"breath_first_search for ele {ele}: {breath_first_search(root,ele)}")

    print(f"tree sum: {tree_sum(root)}")
    print(f"tree_min_value: {tree_min_value(root)}")
    print(f"max_depth_of_tree: {max_depth_of_tree(root)}")


    print(f"\n\narray_representation: {array_representation(root)}")
    array_tree = array_representation(root)

    print(f"breath_first_array: {breath_first_array(array_tree)}")


    print(f"depth_first_array: {depth_first_array(array_tree)}")
    print(f"depth_first_array_recursive: {depth_first_array_recursive(array_tree)}")



if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
