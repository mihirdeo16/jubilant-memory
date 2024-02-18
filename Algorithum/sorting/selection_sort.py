#!/usr/bin/env python3
"""
Selection Sort - with T - O(n^2) and M - O(1)
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

def selection_sort(arr):
    """
    For each element in the array, find the minimum element in the array and swap it with the current
    element
    
    :param arr: The array to be sorted
    :return: The sorted array
    """
    for i in range(0,len(arr)):
        minIdx = i
        for j in range(i+1,len(arr)):
            if arr[minIdx] > arr[j]:
                minIdx = j
        if minIdx != i:
            arr[minIdx],arr[i] = arr[i],arr[minIdx]
    return arr

def main(arr):
    """ Main entry point of the app """

    # Call the selection sort
    arr = selection_sort(arr)

    # Print the results
    print(f'Sorted element using Selection Sort : {arr}')


if __name__ == "__main__":
    """ This is executed when run from the command line """
    print("Provide the input array of the elements")
    IntArray = list(map(int,input().strip().split(" ")))
    main(IntArray)
