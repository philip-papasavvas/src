"""
From: https://github.com/zhiwehu/Python-programming-exercises/blob/master/100%2B%20Python%20challenging%20programming%20exercises.txt
Date: 04/11/2018
"""

#Q1, L1

#Find all numbers divisible by 7 but not multiple of 5
# between 2000 and 3200

lowerBound = 2000
upperBound = 3200

lst = []
for i in range(lowerBound, upperBound+1):
    if (i % 5 != 0) & (i % 7 == 0):
        lst.append(i)


#Q2L1
def factorial(num):
    isinstance(num, int)
        result = 1
        for i in range(2, num+1):
            result *= i
        return result
factorial(6)

# factorial(5.6)
# also
# def fact(x):
#     if x == 0:
#         return 1
#     return x*fact(x-1)
# fact(8.7)


# Q3L1
# Dictionary of integers up to n with their square as the
#value and the key as the integer

def integral(n):
    keys = [i for i in range(n)]
    values = [i**2 for i in range(n)]
    result = dict(zip(keys,values))
    return result

integral(12)

# also could've done
# n = int(input())
# d = dict()
# for i in range(1, n+1):
#     d[i] = i*i
# print(d)

#Q6 L2
# calculate and print value according to formula
# q = sqrt((2CD)/H)
import numpy as np


nums = input()
c = 50
h = 30
value = []
items = [x for x in nums.split(",")]
for i in items:
    value.append(str(int(round(np.sqrt(2*c*float(i)/h)))))
l = ",".join(value)
print(l)

def calc(nums):
    numbers = nums.split(",")
    result = 2*c*d / h
    return np.sqrt(result)