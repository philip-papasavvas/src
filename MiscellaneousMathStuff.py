

def fibonnacci(nums):
    """
    :param nums: Number of parts of the Fibonacci sequence from the beginning
    :return: List of numbers
    """
    i = 1
    if nums <= 1:
        fib = [1]
    elif nums == 2:
        fib = [1,1]
    elif nums > 2:
        fib = [1,1]
        while i < (nums - 1):
            fib.append(fib[i] + fib[i-1])
            i += 1
    return fib

fibonnacci(12)