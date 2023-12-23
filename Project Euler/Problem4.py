import math

# Problem 1
# def sum():
#     sum = 0
#     for i in range(1000):
#         if i % 3 == 0 or i % 5 == 0:
#             sum += i
#     print(sum)

# Problem 10
def sumOfPrimes():
    sum = 0
    for i in range(1, 2000000):
        if isPrime(i):
            sum += i
    print(sum)

def isPrime(n):
    if n < 2:
        return False
    elif n == 2:
        return True
    else:
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

# Problem 20
def factorialDigitSum():
    sum = 0
    for i in str(math.factorial(100)):
        sum += int(i)
    print(sum)

# Problem 25
def fibonacci():
    a = 1
    b = 1
    c = 0
    index = 2
    while len(str(c)) < 1000:
        c = a + b
        a = b
        b = c
        index += 1
    print(index)

# Problem 30
def digiSum():
    sum = 0
    for i in range(1000000):
        if i == 0 or i == 1:
            continue
        digitsum = 0
        for q in str(i):
            digitsum += int(q) ** 5
        if digitsum == i:
            sum += i
    
    print(sum)

# Problem 35
def circularPrimes():
    num = 0
    for i in range(1000000):
        rot = cal(i)
        rotPrime = True
        for q in rot:
            if not isPrime(q):
                rotPrime = False
        if rotPrime:
            num += 1
    print(num)
    
# function to return the count of digit of n
def numberofDigits(n):
    cnt = 0
    while n > 0:
        cnt += 1
        n //= 10
    return cnt
     
# function to print the left shift numbers
def cal(num):
    init = num
    digit = numberofDigits(num)
    powTen = pow(10, digit - 1)
    res = []
     
    for i in range(digit - 1):
         
        firstDigit = num // powTen
         
        # formula to calculate left shift
        # from previous number
        left = (num * 10 + firstDigit -
               (firstDigit * powTen * 10))
        # print(left, end = " ")
        res.append(left)
         
        # Update the original number
        num = left
    res.append(init)
    return res

# Problem 40
def chap():
    const = ""
    for i in range(1000000):
        if len(const) < 1000001:
            const += str(i)
    return const

def chaped():
    const = chap()
    print(int(const[1]) * int(const[10]) * int(const[100])* int(const[1000])* int(const[10000]) * int(const[100000]) * int(const[1000000]))

# Problem 45
def triangle(n):
    return n*(n+1)/2

def pentagon(n):
    return n*(3*n-1)/2

def hex(n):
    return n*(2*n-1)

def triplet():
    t = 286
    p = 166
    h = 144
    while True:
        t += 1
        while pentagon(p) < triangle(t):
            p += 1
        while hex(h) < triangle(t):
            h += 1
        if triangle(t) == pentagon(p) == hex(h):
            print(triangle(t))
            break

# Problem 50
def primeSum():
    primes = [i for i in range(2, 10000) if isPrime(i)]
    max_length = 0
    max_prime = 0
    for i in range(len(primes)):
        for j in range(i + max_length, len(primes)):
            sum_of_primes = sum(primes[i:j])
            if sum_of_primes < 1000000:
                if isPrime(sum_of_primes):
                    max_length = j - i
                    max_prime = sum_of_primes
            else:
                break
    print(f"The prime {max_prime} can be written as the sum of the most ({max_length}) consecutive primes.")

# primeSum()

def isPalindrome(n):
    if (str(n) == str(n)[::-1]):
        return True
    return False

def reversed(n):
    return int(str(n)[::-1])

# Problem 55
def islychrel(n):
    for i in range(50):
        n = n + reversed(n)
        if isPalindrome(n):
            return False
    return True

def countLychrel():
    count = 0
    for i in range(10000):
        if islychrel(i):
            count += 1
    print(count)

