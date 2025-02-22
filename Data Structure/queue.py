#!/usr/bin/env python3
"""
Queue implementation in Python using Custom class and function.
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

class Queue:
    def __init__(self) -> None:
        self.queue = []
    def add(self,var) -> None:
        self.queue.append(var)
    def remove(self) -> None:
        if len(self.queue) != 0:
            val = self.queue.pop(0)
            print(val)
    def peek(self) -> None:
        print(self.queue)

def class_main(operation,operations):
    """ Main entry point of the of Queue class """
    if operation.startswith("add"):
        operation = operation.strip().split(" ")
        operations[operation[0]](operation[1])
    else:
        operations[operation]()

def fun_main(operation,queue):
    """ Main entry point of queue using function """

    if operation.startswith("add"):

        _, value = operation.split() 
        queue.append(value)

    elif operation.startswith("remove"):

        queue.pop(0)
        
    else:
        print(queue)

if __name__ == "__main__":
    """ This is executed when run from the command line """

    q= True

    qu = Queue()
    operations = {
        "add": qu.add,
        "remove": qu.remove,
        "peek": qu.peek
    }

    queue = []

    while q:
        print("Press q to quit or perform add, pop and peek operation: ")

        operation = input()

        if str(operation) == "q":
            exit()

        # class_main(operation,operations)
        fun_main(operation,queue)
