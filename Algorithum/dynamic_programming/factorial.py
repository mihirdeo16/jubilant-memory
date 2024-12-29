#!/usr/bin/env python3
"""
HashMap
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

import time 
def factorial(n)-> int:
    """Recursive call to factorial.

    Args:
        n (int): Input int

    Returns:
        int: return factorials
    """
    if n == 1:
        return 1
    return factorial(n-1)*n

def main(n) -> None:
    """ Main entry point of the app """
    start_time = time.time()
    print(f"N factorial is {factorial(n)}")
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time)* 1000:.4f} seconds")



if __name__ == "__main__":
    """ This is executed when run from the command line """
    n = int(input("Enter the Number :"))
    main(n)
