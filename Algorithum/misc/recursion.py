#!/usr/bin/env python3
"""
Recursion: When developing mindset/logic for recursion do think about following:
1. What is the least amount of work, that can fulfil base case or smallest input case
2. When we would call process complete as return statement.
3. How call stack will work and how its returns will be commutated

Examples of recursion implementation:
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

from typing import Dict, List, Tuple


def reverse_string_fn(input_string: str):
    if input_string == "":
        return ""
    return reverse_string_fn(input_string[1:]) + input_string[0]


def is_palindrome(input_string: str):

    if len(input_string) == 0 or len(input_string) == 1:
        return True

    if input_string[0] == input_string[-1]:

        return is_palindrome(input_string[1:len(input_string)-1])

    return False


def find_binary(input_decimal: int, res=""):

    if input_decimal == 0:
        return res

    res += str(input_decimal % 2)

    return find_binary(input_decimal//2, res)


def sum_natural_numbers(input_decimal: int):
    if input_decimal <= 1:
        return 0

    return input_decimal + sum_natural_numbers(input_decimal-1)


def binary_search(arr, dst, left, right):

    if left > right:
        return -1

    mid = (left+right)//2
    if arr[mid] == dst:
        return mid

    if arr[mid] > dst:
        return binary_search(arr, dst, left, mid-1)
    return binary_search(arr, dst, mid-1, right)


def fibonacci_sequence(n):

    if n <= 1:
        return n

    return fibonacci_sequence(n-1) + fibonacci_sequence(n-2)


def merge_sort(arr):
    """
    https://www.hackerearth.com/practice/algorithms/sorting/merge-sort/visualize/
    """

    if len(arr) > 1:

        left_arr = arr[:len(arr)//2]
        right_arr = arr[len(arr)//2:]

        merge_sort(left_arr)  # This will hit base condition of [1,2] or [2]
        merge_sort(right_arr)

        l, r, s = 0, 0, 0

        while l < len(left_arr) and r < len(right_arr):

            if left_arr[l] >= right_arr[r]:
                arr[s] = right_arr[r]
                r += 1
            else:
                arr[s] = left_arr[l]
                l += 1
            s += 1

        while l < len(left_arr):
            arr[s] = left_arr[l]
            s += 1
            l += 1
        while r < len(right_arr):
            arr[s] = right_arr[r]
            s += 1
            r += 1

    return arr

def link_list_reversal(head):
    if head.val is None or head.next is None:
        return head
    
    temp = link_list_reversal(head)

    temp.next.next = head
    head.next = None

    return temp

def merge_linklists_sorted(head_1,head_2):
    if head_1.val is None:
        return head_1
    if head_2.val is None:
        return head_2
    
    if head_1.val < head_2.val:
        head_1.next = merge_linklists_sorted(head_1.next,head_2)
        return head_1
    else:
        head_1.next = merge_linklists_sorted(head_1,head_2.next)
        return head_2


def insert_binary_search_tree(tree,node):

    # What is least amount of work I should do
    if tree is None:
        return node
    if tree.val > node:
        tree.left = insert_binary_search_tree(tree.left,node)
    else:
        tree.right = insert_binary_search_tree(tree.right,node)
    return tree

def print_leaf_nodes(tree):
    if tree is None:
        return None

    # What is least amount of work I should do
    if tree.left is None and tree.right is None:
        return tree.value
    
    return [print_leaf_nodes(tree.left) + print_leaf_nodes(tree.right)]

def main():
    """ Main entry point of the app """
    input_str = "I am recursion"
    reverse_string = reverse_string_fn(input_str)
    print(reverse_string)

    input_str = "abcba"
    reverse_string = is_palindrome(input_str)
    print(reverse_string)

    input_decimal = 9
    binary_string = find_binary(input_decimal)
    print(binary_string)

    input_decimal = 9
    binary_string = sum_natural_numbers(input_decimal)
    print(binary_string)

    fibonacci_number = fibonacci_sequence(9)
    print(fibonacci_number)

    array = [1, 6, 2, 3, 4]

    merge_sort(array)
    print(array)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
