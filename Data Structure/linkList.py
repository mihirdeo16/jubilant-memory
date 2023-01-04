#!/usr/bin/env python3
"""
LinkList using class/OOP
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

# The Node class is a blueprint for creating objects that have a value and a pointer to the next node.
class Node:
    def __init__(self,val=None,next=None) -> None:
        self.val = val
        self.next = next
    def __repr__(self) -> str:
        return f"Node --> {self.val}"

class LinkList:
# The LinkList class has a head attribute that is initialized to None.
    def __init__(self) -> None:
        self.head = None
    def __repr__(self) -> str:
        """
        We start at the head of the linked list and traverse it until we reach the tail. 
        
        We then print out the value of each node as we traverse the linked list. 
        
        If the node is the head, we print out "Head ->" followed by the value of the node. 
        
        If the node is the tail, we print out "->" followed by the value of the node and then "-> Tail". 
        
        If the node is neither the head nor the tail, we print out "->" followed by the value of the node.
        """
        temp = self.head
        str_ = ""
        while temp:
            if temp is self.head:
                str_ += f"Head -> {temp.val} "
            elif temp.next is None:
                str_ += f"--> {temp.val} -> Tail "
            else:
                str_ += f" --> {temp.val} "

            temp = temp.next
        return str_
    def size(self) -> int:
    # This is a method to count the number of elements in the LinkList.
        temp = self.head
        count = 0
        while temp:

            temp = temp.next
            count +=1
        return count

    def append(self,val):
    # This is the append method.
        if self.head:
            temp = self.head
            while temp.next:
                temp = temp.next
            node_val = Node(val)
            temp.next = node_val
        else:
            self.head = Node(val)
            self.head.next = None

    def prepend(self,val):
    # This is the prepend method.
        new_node = Node(val)
        new_node.next = self.head
        self.head = new_node

    def insert(self,val,position):
        # This is the insert method.
        if position == 0:
            next_pointer = self.head.next
            self.head = Node(val,next_pointer)
        else:
            temp = self.head
            current_position = 0
            while temp:
                current_position +=1 
                if current_position == position:
                    next_pointer = temp.next
                    new_node = Node(val=val,next=next_pointer)
                    temp.next = new_node
                    break
                temp = temp.next

    def delete(self,position):
    # This is the delete method.
        if position == 0:
            self.head = self.head.next
            return self.head       
        else:
            temp = self.head
            current_position = 0
            while temp:
                current_position += 1
                if current_position == position:
                    next_pointer =temp.next.next
                    temp.next = next_pointer
                    break
                temp= temp.next

    def remove(self,val):
    # This is the remove method.
        if self.head.val == val:
            self.head = self.head.next 
        else:
            temp = self.head
            while temp:
                val_pointer = temp.next
                if val_pointer.val == val:
                    next_pointer = temp.next.next
                    temp.next = next_pointer
                    break
                temp = temp.next

    def reverse(self):
    # The above code is reversing the linked list.
        prev,curr = None,self.head
        while curr:
            next_pointer = curr.next
            curr.next = prev
            prev = curr 
            curr = next_pointer
        self.head = prev

    def search(self,val):
    # This is the search method.
        temp = self.head
        current_position = 0
        while temp:
            if temp.val == val:
                return(f"Element {val} is at position {current_position}")
            current_position +=1
            temp = temp.next
        return(f"Element in the LinkList does not exits")
