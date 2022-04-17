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

if __name__=='__main__':
    """To test the this use n = 100 """
    n = int(input("Enter the Number :"))
    print(fibdyn(n))
