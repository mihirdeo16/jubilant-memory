#!/usr/bin/env python3
"""
Bit manipulation

Rules:
+ Understand problem
+ Do Observation/Build logic
+ Make EdgeCase/BaseCases
+ Then write


Q - What is time complexity at when running in loop byte string, 
Ans - O(1) since a byte string is made up of 32 bit
"""

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"


def single_number(nums)-> int:
    """
    Find num which only appears once
    + If number apper even times and we do xor can cancel out
    BaseCase: 
        + What if nums is empty
    T - O(n)
    M - O(1)
    """
    if not nums:
        return 0
    res = 0
    for n in nums:
        res = res ^ n
    return res

def numbers_of_ones(n):
    """
    + Understand problem;
        We want to calculate number of ones in bit reps of number.
    + Do Observation/Build logic
        For given bit check by "&" with bit and if 1 then 1 or 0
        Iterate over by righ tshift 
    + Make EdgeCase/BaseCases
        What if byte string is empty return None
    + Then write -->

    T - O(1) # At max its 32 bit
    M - O(1) # Counter
    """

    count = 0

    while n:
        count += 1 if n & 1 else 0
        n = n >> 1
    return count

def count_bit_range(n):
    """
    + Understand problem
        - We want to count total bit in elements from given range.
    + Do Observation/Build logic
        1) Solution:
            - How to calculate a single number:
                - A number is presented as range of 1s and 0s.
                - To know bit "1" is or not, "&" with 1 if yes then 1 else 0
                - Then do right-shift on number to see all elements of number.
            - Repeat the work in loop. And store in array
            
            T - O(n )
            M - O(n)
            ??
        2) Solution:
            ??

    + Make EdgeCase/BaseCases.
        _ If number is empty or zeros then 0
    + Then write -->
    """

    res = []
    for i in range(1,n+1):
        count = 0
        while i:
            count += 1 if i & 1 else 0
            i >>= 1
        res.append(count)

    return res

def reverse_bit(n):
    """
    Rules:
    + Understand problem
        - We want to reverse byte string
    + Do Observation/Build logic
        - Detect one by using "&".
        - To store it show be at first place, so we can do here left shift.
        - We can do right shift on n to iterate.

    + Make EdgeCase/BaseCases
        - If number is empty return none
    + Then write -->
    """ 
    if n is None:
        return None
    res = 0

    
    pos = 32
    while n:
        bit = 1 if n & 1 else 0
        n = n >> 1
        pos = pos - 1
        res = res | (bit << pos )
    return res

def sum_two_integers(a,b):
    """
    Rules:
    + Understand problem
        Sum the two integer without using + or -, so bit manipulation can be done.
    + Do Observation/Build logic
        - Lets iterates over each bit of these number
        - So if a = 1 xor b = 1 -> pos_res = 0 AND carry = 1  
                a = 1 xor b = 0 -> pos_res = 1 AND carry = 0
                a = 0 xor b = 1 -> pos_res = 1 AND carry = 0
                a = 0 xor b = 0 -> pos_res = 0 AND carry = 0 
                    temp_res = 0 OR carry = 1  -> res = 1
                    temp_res = 0 OR carry = 0  -> res = 0
                    temp_res = 1 OR carry = 1  -> res = 1
                    temp_res = 1 OR carry = 0  -> res = 1
        - Conclusion: To get pos_res from a and b use XoR; and to get carry use AND. final_pos do carry OR pos_res.
        - Store and pos 

        - Finally add bit to final, and left shift 
    + Make EdgeCase/BaseCases Only if a or b is None
    + Then write -->

    T - O(1); M - O(1)
    """

    res = 0
    carry = 0
    pos = 0
    while a:

        pos_res = a ^ b
        res_bit = pos_res | carry

        carry = a & b

        res = res | (res_bit << pos)
        pos += 1

        a >>= 1 # Image this: To which direction keep nose that direction.
        b >>= 1
    return res 


def main():
    """ Main entry point of the app """
    nums = [2,2,1]
    nums = [4,1,2,1,2]
    nums = [1]
    res = single_number(nums)
    print(f"single_number: {res}")

    n = 0o1111111111111111111111111111101
    res = numbers_of_ones(n)
    print(f"numbers_of_ones: {res}")

    n = 0o000000000000000000000000000001 # 1 to 2147483648
    res = reverse_bit(n)
    print(f"reverse_bit: {res}")

    a, b = 4 , 5
    res = sum_two_integers(a,b)
    print(f"sum_two_integers: {res}")

    n = 10
    res = count_bit_range(n)
    print(f"count_bit_range: {res}")
    
    
if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
