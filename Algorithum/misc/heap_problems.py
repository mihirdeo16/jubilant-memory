#!/usr/bin/env python3
"""
Heap Problems

Rules:
+ Understand problem
+ Do Observation/Build logic
+ Make EdgeCase/BaseCases
+ Complexity


minHeap: (Python in-build module) Where root is Smallest, POP will remove root (smallest element).
maxHeap: (Use using -ve value as key) Where root is Largest, POP will remove root (largest element).
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

import heapq
from typing import List

class KthLargest:
    """
    Rules:
    + Understand problem
    + Do Observation/Build logic
        - Need something which handle dynamic insertion while keeping track of element order.
    + Make EdgeCase/BaseCases
    + Then write

    """

    def __init__(self, k: int, nums: List[int]):
        self.minHeap = nums
        self.k = k

        heapq.heapify(self.minHeap)
        while len(self.minHeap) > self.k:
            heapq.heappop(self.minHeap)

    def add(self, num: int):
        heapq.heappush(self.minHeap,num)
        while len(self.minHeap) > self.k:
            heapq.heappop(self.minHeap)
        return self.minHeap[-1]



def last_stone_weights(stones):
    """
    Rules:
        + Understand problem
            - Have to take two stone which are heaviest and smash them utile only one remain.
        + Do Observation/Build logic
            - Need to sort to access largest two.
            - Check if smash result into zero then nothing
            - If smash result into something, then add back as element.
            - Indicate dynamic sorting, 
            - Stop until only one remain
        + Make EdgeCase/BaseCases
            - if no stone left then return 0
            - if stone are empty then return 0
            - if stone is only one then return stone[0]
        + Complexity
            T - O(n)
            M - O(n)
    """

    if not stones:
        return 0

    stones = [-ele for ele in stones]

    heapq.heapify(stones)

    while len(stones) > 1:
        x = -1 * heapq.heappop(stones)
        y = -1 * heapq.heappop(stones)

        if x == y:
            continue
        elif x > y:
            diff = x - y
            heapq.heappush(stones, -diff)
        else:
            diff = y - x
            heapq.heappush(stones, -diff)

    return -1 * stones[0] if stones else 0


class MedianFinder:
    """
    Rules:
        + Understand problem
            - We want to find median values in steaming data.
            - One method will add and one will ask for median value.
        + Do Observation/Build logic
            1) Solution:
                - Just append in array addNum
                - Sort every time when asked for medianFinder 
                k - number of time findMedian called; n - number of elements
                T - O(n * log n * k) 
                M - O(n)

            2) Solution: 
                j - number of time addNum called
                T - O(log n * j) & M - O(n)
                Approach:
                    AddNum:
                        1. Create two heaps which track smallElems and LargeEle compared to current one.
                        2. For smallElements it is maxHeap; means root is largest.
                        3. For largeElements it is minHeap; means root is smallest.
                        4. Push element to smallElements if ele < root else push ot LargeElement
                        5. Rebalance heap, with only allow to exceed length by one w.r.t other.
                    findMedian:
                        1. Compared sizes if same; take roots and calculate median.
                        2. Else check which has larger size take his root.
                    Note: Python has only minHeap which mean need to reverse smallElement heap.

        + Make EdgeCase/BaseCases
            - If findMedian called before any of heap has elements raise issue/
        + Complexity

    """

    def __init__(self):
        self.largeElem = []
        self.smallElem = []

    def addNum(self, ele):
        if self.smallElem == [] or self.largeElem == []:
            if self.smallElem == []:
                heapq.heappush(self.smallElem, -ele)
            else:
                heapq.heappush(self.largeElem, ele)
        else:
            if ele > self.largeElem[0]:
                heapq.heappush(self.largeElem, ele)

                # Check for rebalance of Heap
                while len(self.smallElem) < len(self.largeElem) - 1:
                    ele = heapq.heappop(self.largeElem)
                    heapq.heappush(self.smallElem, -ele)
            else:
                heapq.heappush(self.smallElem, -ele)

                # Check for rebalance of Heap
                while len(self.largeElem) < len(self.smallElem) - 1:
                    ele = -heapq.heappop(self.largeElem)
                    heapq.heappush(self.smallElem, ele)

    def findMedian(self) -> float:
        if self.largeElem==[] and self.smallElem==[]:
            raise ValueError("Not allowed to perform findMedian before atleast one addNum operation")
        if len(self.largeElem) == len(self.smallElem):
            return (-self.smallElem[0] + self.largeElem[0]) / 2
        return self.largeElem[0] if len(self.largeElem) > len(self.smallElem) else -self.smallElem[0]


def main():
    """ Main entry point of the app """

    stones = [2, 7, 4, 1, 8, 1]
    res = last_stone_weights(stones)
    print(f"last_stone_weights: {res}")

    # Class implementation
    # kthLargest = KthLargest(3, [4, 5, 8, 2]);
    # print(kthLargest.add(3))   # return 4
    # print(kthLargest.add(5))   # return 5
    # print(kthLargest.add(10))  #  return 5
    # print(kthLargest.add(9))   # return 8
    # print(kthLargest.add(4))   # return 8

    medianFinder = MedianFinder()
    # print(medianFinder.findMedian())  # raise issue
    medianFinder.addNum(3)  # arr = [3]
    print(medianFinder.findMedian())  # return 3
    medianFinder.addNum(4)  # arr = [3, 4]
    print(medianFinder.findMedian())  # return 3.5 
    medianFinder.addNum(9)   # arr[3, 4, 9]
    print(medianFinder.findMedian())  # return 4
    medianFinder.addNum(2)
    medianFinder.addNum(1)
    medianFinder.addNum(6)
    medianFinder.addNum(7)   # arr[1, 2, 3,4,6,7,9]
    print(medianFinder.findMedian())  # return 2.0


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
