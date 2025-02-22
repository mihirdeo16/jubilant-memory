#!/usr/bin/env python3
"""
Trie/Prefix Tree implementation: Using HashMap

TODO:
+ Create OOP class of Trie for words : https://youtu.be/qA8l8TAMyig?si=tRm0dqsEndmDN83W
+ Implementation of Trie https://youtu.be/8mhw5WT2x0U?si=LOD44FcNmYmEa5r-
+ Word Search https://youtu.be/asbcE9mZz_U?si=976J_vEMbXs2p1BV
+ https://youtu.be/BTf05gs_8iU?si=dvmtINpmRPBKSWVy
+ https://youtu.be/SG9CPplNGgo?si=dnosQJZX8qHffoiD

"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

from collections import defaultdict

class TrieNode:
    def __init__(self):
        self.childerns = {}
        self.sentences = defaultdict(int)


def main():
    """ Main entry point of the app """
    print("hello world")


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()