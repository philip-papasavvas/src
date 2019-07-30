"""
Author: Philip P
03/11/2018
"""

# def boxPrint(symbol, width, height):
#     if len(symbol) != 1:
#         raise Exception("'symbol' needs to be of length 1.")
#     if (width < 2) or (height < 2):
#         raise Exception('"width" and "height" must be '
#                         'greater than or equal to 2')
#     print(symbol * width)
#
#     for i in range(height - 2):
#         print(symbol + (" " * (width -2)) + symbol)
#
#     print(symbol * width)
#
# # boxPrint('&', 10, 3)
# # boxPrint("£", 10, 15)
# # boxPrint("*", 4,1)
#
# market_2nd = {'ns': 'green', 'ew':'red'}
#
# def switchLights(intersection):
#     for key in intersection.keys():
#         if intersection[key] == 'green':
#             intersection[key] = 'yellow'
#         elif intersection[key] == 'yellow':
#             intersection[key] = 'red'
#         elif intersection[key] == 'red':
#             intersection[key] = 'green'
#     assert 'red' in intersection.values(), \
#     'Neither light is red!' + str(intersection)
#
# print(market_2nd)
# switchLights(market_2nd)
# print(market_2nd)
#
#
# ################
# import logging
# logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s - \
#     %(levelname)s - %(message)s')
# logging.disable(logging.CRITICAL)
#
# logging.debug("Start of program")
#
# def factorial(n):
#     logging.debug("Start of factorial (%s)" % (n))
#     total = 1
#     for i in range(1, n+1):
#         total *= i
#         logging.debug("i is %s, total is %s" % (i, total))
#
#     logging.debug("Return value is %s" % (total))
#     return total
#
# print(factorial(5))
#
# logging.debug("End of program")

####
print("Enter the first number to add:")
first = input()
print("Enter the second number to add:")
second = input()
print("Enter the third number to add:")
third = input()
print("The result is " + first + second + third)
