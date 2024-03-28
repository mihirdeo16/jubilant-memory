#!/usr/bin/env python3

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

def merge_sort(arr):
    """
    Merge Sort - with T - O(n*log(n)) and M - O(1) using Divide and Conquer approach

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

def bubble_sort(arr):
    """
    Bubble Sort - with T - O(n^2) and M - O(1)

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

def selection_sort(arr):
    """
    Selection Sort - with T - O(n^2) and M - O(1)
    
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
    
    # Call the bubble sort
    arr = bubble_sort(arr)

    # Call the merge sort 
    arr = merge_sort(arr)

    # Print the results
    print(f'Sorted element using Bubble Sort : {arr}')

if __name__ == "__main__":
    """ This is executed when run from the command line """
    print("Provide the input array of the elements")
    IntArray = list(map(int,input().strip().split(" ")))
    main(IntArray)