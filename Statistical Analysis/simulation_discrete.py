#!/usr/bin/env python3
"""
HashMap
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

import numpy as np

def sample_gen(uni_eles,probabilities_elems,sample_size) -> list:
    """
    > The function takes in a list of unique elements, a list of probabilities of those elements, and a
    sample size, and returns a list of elements sampled from the unique elements list with the
    probabilities of the elements in the probabilities list
    
    :param uni_eles: list of unique elements in the population
    :param probabilities_elems: The probabilities of each element in the population
    :param sample_size: The number of elements you want to generate
    :return: A list of random numbers
    """
    np.random.seed(123)
    rnt = np.random.default_rng()
    sample = rnt.choice(uni_eles,size=sample_size,p=probabilities_elems)
    return sample

def lottery_win():
    """
    It generates a sample of size 1000 from the word "big" with the given probabilities, and then it
    generates 30 samples from the sample of size 1000. It then counts the number of tickets bought until
    the word "big" is generated
    :return: The number of tickets bought to win the lottery.
    """
    word = ['b','i','g']
    sample_size = 1000
    probabilities_elems = [0.60,0.30,0.10]
    samples = sample_gen(word,probabilities_elems,sample_size)
    n = 30

    ticket_bought  = 0
    check = [0,0,0]
    count_ticket = np.array([])
    
    for _ in range(n):

        for sample in samples:
            ticket_bought +=1

            if sample =="b":
                check[0] = 1
            elif sample =="i":
                check[1] =1
            else:
                check[2]=1
                
            if sum(check)==3:
                count_ticket = np.append(count_ticket,ticket_bought)
                break
    return count_ticket

def caramel_boxes():
    """
    It generates a sample of size 1000, and then for each sample, it checks if the sample contains all
    the prizes. If it does, it stores the number of boxes bought in an array
    :return: The number of boxes bought to get all 4 prizes.
    """
    prizes = ["A","B","C","D"]
    probability = [0.25,0.25,0.25,0.25]
    sample_size = 1000
    samples = sample_gen(prizes,probability,sample_size)

    n = 40
    box_bought = 0
    check = [0,0,0,0]
    count_boxes = np.array([])

    for _ in range(40):
        for sample in samples:
            box_bought +=1 
            if sample == "A":
                check[0]=1
            elif sample == "B":
                check[1]=1            
            elif sample == "C":
                check[2]=1            
            else:
                check[3]=1            

            if sum(check)==4:
                count_boxes = np.append(count_boxes,box_bought)
                break
    return count_boxes

def main():
    """ Main entry point of the app """

    
    """
    To win a certain lotto, a person must spell the word big. Sixty percent of the tickets contain the letter b, 30% contain the letter i, and 10% contain the letter g. Find the average number of tickets a person must buy to win the prize. Run for 30 times this experiements
    """
    count_ticket = lottery_win()
    print("Average number of tickets a person must buy to win the prize :",np.mean(count_ticket))

    """
        A caramel corn company gives four different prizes, one in each box. They are placed in the boxes at random. Find the average number of boxes a person needs to buy to get all four prizes. (40) 
    """
    count_boxes = caramel_boxes()
    print("Average number of boxes a person needs to buy to get all four prizes :",np.mean(count_boxes))


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
