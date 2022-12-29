#!/usr/bin/env python3
"""
Stack implementation in Python using Custom class and function.
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

class Stack:
    """Stack class """
    
    def __init__(self) -> None:
        self.stack = []
    def push(self,var) -> None:
        self.stack.append(var)
    def pop(self) -> None:
        if len(self.stack) != 0:
            val = self.stack.pop(-1)
            print(val)
    def peek(self) -> list:
        print(self.stack)

def class_main(operation,operations):
    """ Main entry point of the of Queue class """

    if operation.startswith("push"):
        operation = operation.strip().split(" ")
        operations[operation[0]](operation[1])
    else:
        operations[operation]()

def fun_main(operation,stack):
    """ Main entry point of queue using function """

    if operation.startswith("push"):

        _, value = operation.split() 
        stack.append(value)

    elif operation.startswith("pop"):

        val = stack.pop(-1)

        print(val)
        
    else:
        print(stack)

if __name__ == "__main__":
    """ This is executed when run from the command line """

    q= True
    stk = Stack()

    operations = {
        "push": stk.push,
        "pop": stk.pop,
        "peek": stk.peek
    }

    stack = []

    while q:
        print("Press q to quit or perform add, pop and peek operation: ")

        operation = input()

        if str(operation) == "q":
            exit()

        class_main(operation,operations)
        # fun_main(operation,stack)
