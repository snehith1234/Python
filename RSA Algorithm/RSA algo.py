import random
#extended GCD calculator
def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    else:
        s = b % a
        g, x, y = extended_gcd(s, a)
        p = b//a
        return (g, y - (p) * x, x)
def mod_inverse(phi_n, e):
        g, x, _ = extended_gcd(phi_n, e)
        while g != 1:
                e = random.randint(2,(phi_n-1))
                g, x, _ = extended_gcd(phi_n,e)
        e  = e % phi_n
        for x in range(1, phi_n) :
                if ((e * x) % phi_n == 1) :
                        return x
        return 1
def generate(n):
    while prime(n) == False:
        n = random.randint(1,100)
    return n
    
def prime(n):
    n = int(n)
    if n < 2:
        return False
    if n == 2: 
        return True    
    if not n & 1:
        return False
    for x in range(3, int(n**0.5)+1, 2):
        if n % x == 0:
            return False
    return True
p = random.randint(1,100)
p = generate(p)
q = random.randint(1,100)
q = generate(q)
num = p * q
phi_n = (p-1)*(q-1)
e = random.randint(2,(phi_n)-1)
e = 3
print p,q
pk = mod_inverse(phi_n, e)
print "Public key::",(num,e)
print "Private key::",pk


    
