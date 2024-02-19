#!/usr/bin/env python3
"""
TF-IDF implementation
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"



import numpy as np
from typing import List
import re
from collections import defaultdict

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