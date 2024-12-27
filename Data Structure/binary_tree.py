#!/usr/bin/env python3
"""
Binary Tree implementation in Python using Custom class and function.
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"


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


def main():
    """
    Sample Tree:
          3
        /   \
       11      4
      /  \       \
     4    2       1

    """

    leftChild_2_1 = Node(4)
    rightChild_2_1 = Node(2)
    rightChild_2_2 = Node(1)

    leftChild = Node(11, left=leftChild_2_1, right=rightChild_2_1)
    rightChild = Node(4, right=rightChild_2_2)
    root = Node(3, left=leftChild, right=rightChild)

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


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
