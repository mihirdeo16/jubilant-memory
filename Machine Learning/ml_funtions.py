#!/usr/bin/env python3
"""
ML function implemented using core python and Numpy
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

from collections import defaultdict
import numpy as np
from typing import List
import re

def cosine_similarity(a,b) -> np.float64:
    return np.dot(a,b)/ (np.sqrt(np.sum(np.square(a)))*np.sqrt(np.sum(np.square(b))))

def euclidean_distance(a,b) -> np.float64:
    return np.sqrt(np.sum(np.square(b-a)))

class TFIDF:
    """TFIDF Implementation with custom logic
    """

    def __init__(self) -> None:
        self.N = 0
        self.word_dict = defaultdict(lambda: defaultdict(int))

    def transform(self,text_list:List[str]) -> np.ndarray:
        return np.array([self.apply(sentence) for sentence in text_list])
    
    def fit_transform(self,text_list:List[str]) -> np.ndarray:
        self.fit(text_list)
        return self.transform(text_list)
    
    def fit(self,text:List[str])-> None:
        self.N = len(text)
        text = [re.sub(r'[^\w\s]', '', sen.lower())  for sen in text ]

        for sentence in text:
            for word in sentence.split():
                if self.word_dict.get(word,0):
                    self.word_dict[word]['tf'] += 1
                else:
                    self.word_dict[word]['tf'] = 1
            for word in self.word_dict:
                if word in sentence:
                    self.word_dict[word]['df'] += 1

    def apply(self, sentence: str) -> np.ndarray: 

        sentence = re.sub(r'[^\w\s]', '', sentence.lower()) 

        word_vector = []
        for word in sentence.split():
            if self.word_dict[word]['df'] == 0:
                word_val = 0
            else:
                idf = self.N / self.word_dict[word]['df']
                word_val = self.word_dict[word]['tf'] * np.log(idf)
            word_vector.append(word_val)
        return np.array(word_vector)
    


def main() -> None:
    """ Main entry point of the app """
    vector_1 = np.array([3,6,1,7,4,8])
    vector_2 = np.array([6,1,4,6,2,6])
    cosine_val = cosine_similarity(vector_1,vector_2)
    euclidean_val = euclidean_distance(vector_1,vector_2)

    print(f"Cosine similarity: {cosine_val}")
    print(f"Euclidean distance: {euclidean_val}")

    corpus = [
        'This is the first first first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?'
        ]
    tf_idf_obj = TFIDF()
    tf_idf_obj.fit(corpus)
    tfidf_vec = tf_idf_obj.transform(["This is the first first first document."]) # tf_idf_obj.fit_transform(corpus)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()