#!/usr/bin/env python3
"""
Correlation 
"""
import math
__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

def mean(array):
    return sum(array)/len(array)

def variance(array,mean):
    num = [(x-mean)**2 for x in array]
    return sum(num)/(len(array))

def standard_deviation(variance_value):
    return math.sqrt(variance_value)

def covariance(result):
    num = 0
    for x,y in zip(result[0]["values"],result[1]["values"]):
        num += (x-result[0]["mean"])*(y-result[1]["mean"])
    return num/ (len(result[0]["values"]))


def correlation(result):
    return result["covariance"]/(result[0]['std']*result[1]['std'])

def main(x,y):
    """ Main entry point of the app """

    result = {}

    for ind,array in enumerate([x,y]):

        mean_value = mean(array)
        variance_value = variance(array,mean_value)
        std_value = standard_deviation(variance_value)
        result[ind] = {
            "values":array,
            "mean":mean_value,
            "variance":variance_value,
            "std":std_value
        }
    result["covariance"] = covariance(result) 
    result["correlation_value"]  = correlation(result)
    print(result)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    x = list(map(int,input().strip().split()))
    y = list(map(int,input().strip().split()))
    
    # x = [1,2,3,4,5,6,7,8,9,10]
    # y = [10,9,8,7,6,5,4,3,2,1]

    main(x,y)


