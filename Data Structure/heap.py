#!/usr/bin/env python3
"""
Heap implementation, there are different types of heaps, we are focusing on Heap: minHeap and maxHeap,

Time complexity of the operations:

To extract minimum/maximum element: O(1)
To build heap, know as Heapify: O(n)
To push the element in heap, Push: O(log n)
To push the element in heap, Pop: O(log n)

"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

import heapq
from typing import Dict, List, Tuple


class MaxHeap:

    """
    Access element by:
    
    parent:(i-1)//2, 
    left:  2*i + 1 
    right: 2*i + 2


    """

    def __init__(self) -> None:
        self.heap = []

    def display(self):
        print(self.heap)

    def build(self, array):
        """
        Time complexity: O(n)
        """
        for ele in array:
            self.insert(ele)

    def insert(self, ele):
        """
        Time complexity: O(log n) due to siftup
        """
        self.heap.append(ele)
        self.siftup(len(self.heap)-1)

    def siftup(self, i):
        """
        Time complexity: O(log n) 
        """
        parents = (i-1)//2

        # To get max element at top
        while i != 0 and self.heap[i] > self.heap[parents]:
            self.heap[parents], self.heap[i] = self.heap[i], self.heap[parents]
            i = parents
            parents = (i-1)//2

    def sifdown(self, i):
        """
        Time complexity: O(log n) 
        """
        left_child_idx = 2*i + 1
        right_child_idx = 2*i + 2

        while (left_child_idx < len(self.heap) and self.heap[i] < self.heap[left_child_idx]) or (right_child_idx < len(self.heap) and self.heap[i] < self.heap[right_child_idx]):

            if self.heap[left_child_idx] > self.heap[right_child_idx]:
                self.heap[left_child_idx], self.heap[i] = self.heap[i], self.heap[left_child_idx]
                i = left_child_idx

            else:
                self.heap[right_child_idx], self.heap[i] = self.heap[i], self.heap[right_child_idx]
                i = right_child_idx

            left_child_idx = 2*i + 1
            right_child_idx = 2*i + 2

    def get_top(self):
        """
        Time complexity: O(1) 
        """
        if self.heap:
            return self.heap[0]
        return None

    def extract(self):
        """
        Time complexity: O(log n) 
        """
        if self.heap is None:
            return None
        self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]
        max_ele = self.heap.pop()
        self.sifdown(0)
        return max_ele

        


class MinHeap:
    """
    Access element by:
    
    parent:(i-1)//2, 
    left:  2*i + 1 
    right: 2*i + 2
    
    """
    def __init__(self) -> None:
        self.heap = []

    def display(self):
        print(self.heap)

    def build(self, array):
        """
        Time complexity: O(n)
        """
        for ele in array:
            self.insert(ele)

    def insert(self, ele):
        """
        Time complexity: O(log n) due to siftup
        """
        self.heap.append(ele)
        self.siftup(len(self.heap)-1)

    def siftup(self, i):
        """
        Time complexity: O(log n)
        """
                
        parent = (i-1)//2
        while i != 0 and self.heap[i] < self.heap[parent]:

            self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]

            i = parent
            parent = (i-1)//2

    def siftdown(self, i):
        """
        Time complexity: O(log n) 
        """
        left_index = 2*i + 1  
        right_index = 2*i + 2

        while (left_index < len(self.heap) and self.heap[i] > self.heap[left_index]) or (right_index < len(self.heap) and self.heap[i] > self.heap[right_index]):

            if self.heap[right_index] > self.heap[left_index]:
                self.heap[i], self.heap[left_index] = self.heap[left_index], self.heap[i]
                i = left_index
            else:
                self.heap[i], self.heap[right_index] = self.heap[right_index], self.heap[i]
                i = right_index

            left_index = 2*i + 1
            right_index = 2*i + 2

    def extract(self):
        """
        Time complexity: O(log n) due to siftdown
        """
        if self.heap is None:
            return None

        self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]

        min_ele = self.heap.pop()

        self.siftdown(0)

        return min_ele

    def get_top(self):
        """
        Time complexity: O(1)
        """
        if self.heap is None:
            return None
        return self.heap[0]


def main():
    """ Main entry point of the app 
                  3
            /           \
          5              8
        /   |           /
     17     14         12
    """

    # Array to given
    array = [12, 14, 8, 17, 5, 3]

    # Using class implementation
    heap_obj = MinHeap()

    heap_obj.build(array)
    heap_obj.display()

    heap_obj.insert(20)
    heap_obj.insert(10)
    heap_obj.display()

    ele = heap_obj.get_top()
    print("Top ele:", ele)

    heap_obj.extract()
    heap_obj.display()

    sorted_array = [heap_obj.extract() for _ in range(len(array))] # O(n*log n)
    print(sorted_array)

    # # Using python's module
    # heapq.heapify(array) # O(n)
    # print(array)

    # heapq.heappush(array,2) # O(log n)
    # print(array)

    # heapq.heappop(array) # O(log n)
    # print(array)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
