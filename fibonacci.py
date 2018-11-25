import math

def fibonnacci(nums):
    i = 1
    if num <= 1:
        fib = [1]
    elif num == 2:
        fib = [1,1]
    elif num > 2:
        fib = [1,1]
        while i < (nums - 1):
            fib.append(fib[i] + fib[i-1])
            i += 1
    return fib

fibonnacci(12)