#!/usr/bin/env python3
"""
Bubble Sort - with T - O(n^2) and M - O(1)
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

def bubble_sort(arr):
    """
    For each element in the array, find the minimum element in the array and swap it with the current
    element
    
    :param arr: The array to be sorted
    :return: The sorted array
    """
    for i in range(0,len(arr)):
        for j in range(0,len(arr)-i-1):
            if arr[j] > arr[j+1]:
                arr[j],arr[j+1] = arr[j+1],arr[j]
    return arr

def main(arr):
    """ Main entry point of the app """

    # Call the selection sort
    arr = bubble_sort(arr)

    # Print the results
    print(f'Sorted element using Bubble Sort : {arr}')


if __name__ == "__main__":
    """ This is executed when run from the command line """
    print("Provide the input array of the elements")
    IntArray = list(map(int,input().strip().split(" ")))
    main(IntArray)
