# GridTraveler Problem,
# You begin in top left traveler on 2D grid
# and move only bottom-right corner
# No. of ways to reach the goal with deminesion m,n

def gridTravelerDyn(m, n, memo={}):
    '''
    Time: O(m+n)
    Space:O(m+n)
    '''
    key = str(m)+',0'+str(n)
    if (key in memo):
        return memo[key]
    if(m == 1 and n == 1):
        return 1
    if(m == 0 or n == 0):
        return 0
    memo[key] = gridTravelerDyn(m-1, n, memo)+gridTravelerDyn(m, n-1, memo)
    return memo[key]

if __name__=='__main__':
    """To test the this use m n = 100 """
    m, n = map(int, input("Enter the Numbers: m  n : ").strip().split())
    print(f'No. of ways are {gridTravelerDyn(m, n)}')
