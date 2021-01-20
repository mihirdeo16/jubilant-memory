# Fibonacci Sequence Recurssion code ##

def fib(n):
    '''
    It take one parameter as "n" which give back 
    the nth value fibonacci squence.
    Time : O(2**n)
    Space: O(n)
    '''
    if n < 2:
        return 1
    return fib(n-1)+fib(n-2)


print("Enter the Number :")
n = int(input())
print(fib(n))
