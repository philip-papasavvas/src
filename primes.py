def isPrime(num):
    if num >= 2:
        #i = 1
        for k in range(2, num):
            if (num % k) == 0:
                print(str(num) +" is not a prime")
            else:
                print(str(num) + " is prime")

isPrime(11)

def testPrime(n):
    if n == 1:
        return False
    if n == 2:
        return True
    else:
        for x in range(2,n):
            if (n % x) == 0:
                return False
        return True

testPrime(14)


def listCompPrime(num):
    if num <=1:
        print("There are no primes")
    if num >= 2:
        a = [(num % i) for i in range(2,num)]
        lst = []
        if 0 in a:
            print(str(num) + " is not prime")
        else:
            lst.append(num)
            return (lst)
            print(str(num) + " is prime")
listCompPrime(12)

#one liner
#top = int(input())
def returnPrimes(top):
    return([num for num in range(2,top) \
            if all((num % i != 0) for i in range(2,num))])

start = time.time()
returnPrimes(10000)
end = time.time()
end - start


# def primes(num)
#     if num == 0:
#         return "Error"
#     if num == 1:
