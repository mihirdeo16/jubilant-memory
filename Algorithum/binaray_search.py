#!/usr/bin/env python3
"""
Binary Search - Time - O(log(n)), Memory - O(1)
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

def binary_search(array,ele):
    """
    We start with the left and right pointers at the beginning and end of the array, respectively. We
    then find the middle element of the array and compare it to the element we're searching for. If the
    middle element is greater than the element we're searching for, we move the right pointer to the
    middle element minus one. If the middle element is less than the element we're searching for, we
    move the left pointer to the middle element plus one. If the middle element is equal to the element
    we're searching for, we return the index of the middle element. If the left pointer is greater than
    the right pointer, we return "None"
    
    :param array: the array to search through
    :param ele: The element we're searching for
    :return: The index of the element in the array.
    """
    left,right = 0,len(array)-1

    while left <= right:
        mid = (left+right)//2
        if array[mid]>ele:
            right = mid -1
        elif array[mid]< ele:
            left = mid + 1
        else:
            return mid
    return "None"


def main(array,ele):
    """ Main entry point of the app """
    pos = binary_search(array,ele)

    print(f"Element is found at: {pos}")


if __name__ == "__main__":
    """ This is executed when run from the command line """
    array = [1,2,3,4,5,6]

    print(f"Array: {array}")
    ele = int(input("Element to search : "))

    main(array,ele)
