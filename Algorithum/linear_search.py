#!/usr/bin/env python3
"""
Linear Search Time complexity - O(n) Memory Complexity - O(1)
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"


def linear_search(array,ele):
    """
    It takes an array and an element as input and returns the position of the element in the array if it
    is present, else returns "None"
    
    :param array: The array to search through
    :param ele: The element to be searched for
    :return: The position of the element in the array.
    """
    for pos,element in enumerate(array):
        if ele == element:
            return pos
    return "None"

def main(*args):
    """ Main entry point of the app """
    pos = linear_search(array,ele)

    print(f"Element is found at: {pos}")


if __name__ == "__main__":
    """ This is executed when run from the command line """
    array = [1,2,3,4,5,6]

    print(f"Array: {array}")
    ele = int(input("Element to search : "))

    main(array,ele)