#!/usr/bin/env python3
"""
Merge Sort - with T - O(n*log(n)) and M - O(1) using Divide and Conquer approach
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"


def merge_sort(arr):
    """
    Merge_sort(arr, l, r) sorts the array arr[l..r] using merge sort
    
    :param arr: The array to be sorted
    :param l: left index of the sub-array of arr to be sorted
    :param r: the right index of the sub-array of arr we are sorting
    """ 
    if len(arr)>1:
        l_arr = arr[:len(arr)//2]
        r_arr = arr[len(arr)//2:]

        merge_sort(l_arr)
        merge_sort(r_arr)

        # Merge
        l,r,s= 0,0,0

        while l < len(l_arr) and r < len(r_arr):

            if l_arr[l] <= r_arr[r]:
                arr[s] = l_arr[l]
                l +=1 
            else:
                arr[s] = r_arr[r]
                r +=1 
            s +=1

        while l < len(l_arr):
            arr[s] = l_arr[l]
            l +=1
            s +=1
        while r < len(r_arr):
            arr[s] = r_arr[r]
            r +=1
            s +=1
    return arr

def main(arr):
    """ Main entry point of the app """

    # Call the selection sort
    arr = merge_sort(arr)

    # Print the results
    print(f'Sorted element using Merge Sort : {arr}')


if __name__ == "__main__":
    """ This is executed when run from the command line """
    print("Provide the input array of the elements")
    IntArray = list(map(int,input().strip().split(" ")))
    main(IntArray)
