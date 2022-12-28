#!/usr/bin/env python3
"""
HashMap implementation in Python using Dict data type
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"


def main(arr):
    """ Main entry point of the app """
    hashMapCounter = {}
    hashMapIdxF = {}
    hashMapIdxL = {}
    hashMapIdxs = {}

    # HashMapCounter with element  count as value.
    for i in arr:
        if hashMapCounter.get(i,0):
            hashMapCounter[i] += 1
        else:
            hashMapCounter[i] = 1

    print("HashMap counter is : ",hashMapCounter)


    # HasMapIdxF with element index's last occurred value.
    for idx, i in enumerate(arr):
        if hashMapIdxF.get(i,0) == 0:
            hashMapIdxF[i] = idx

    print("HashMap ID with first occurrence is : ",hashMapIdxF)

    # HasMapIdxL with element index's last occurred value.
    for idx, i in enumerate(arr):
        if hashMapIdxL.get(i,0):
            hashMapIdxL[i] = idx
        else:
            hashMapIdxL[i] = idx

    print("HashMap ID with last occurrence is : ",hashMapIdxL)

    # With last element occurrence
    for idx, i in enumerate(arr):
        if hashMapIdxs.get(i,0):
            hashMapIdxs[i].append(idx)
        else:
            hashMapIdxs[i] = [idx]

    print("HashMap ID with index values occurrence is : ",hashMapIdxs)

if __name__ == "__main__":
    """ This take the data in arr and pass to main  function. Following three types of HashMap is created.
    + HashMapCounter with element  count as value.
    + HashMapIdxF with element index's first occurred value.
    + HasMapIdxL with element index's last occurred value.
    + hashMapIdxs with element indexes occurred value """

    # Test = 1 2 3 4 4 
    print("Enter the arr element for HashMap :")
    arr = list(map(int,input().strip().split(" ")))
    main(arr)
    
