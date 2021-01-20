# Fibonacci Sequence using Dynamic Programming code ##

def fibdyn(n, memo={}):
    '''
    It take one parameter as "n" which give back 
    the nth value fibonacci squence. memo (memory) hold the
    memory object
    Time : O(n)
    Space: O(n)
    '''
    if n < 2:
        return 1
    if (n in memo):
        return memo[n]
    memo[n] = fibdyn(n-1, memo)+fibdyn(n-2, memo)
    return memo[n]


print("Enter the Number :")
n = int(input())
print(fibdyn(n))
