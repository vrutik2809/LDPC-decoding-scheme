import numpy as np
def bsc(m,p):
    n = np.array(np.random.rand(1,len(m))).flatten()
    for i in range(len(n)):
        if n[i] < p:
            n[i] = 1
        else:
            n[i] = 0
    n = n.astype(dtype = int)
    r = np.bitwise_xor(m,n)
    return r

def bec(m,p):
    r = bsc(m,p)
    for i in range(len(r)):
        if r[i] == 1:
            r[i] = -1 #Here -1 represents an error
    return r

def locationOfOnes_raw(v):
    arr = []
    for i in range(len(v)):
        if v[i] == 1:
            arr.append(i)
    return np.array(arr)

def locationOfOnes_col(c):
    arr = []
    for i in range(len(c)):
        if c[i] == 1:
            arr.append(i)
    return np.array(arr)

def majorityDecoding_c_hat(P_c,L_c,r):
    count1 = 0
    count0 = 0
    if r == 1:
        count1 += 1
    else:
        count0 += 1

    for i in P_c:
        if L_c[i] == 1:
            count1 += 1
        else:
            count0 += 1

    if count1 > count0:
        return 1
    elif count1 < count0:
        return 0
    else:
        return int(np.random.randint(2,size = 1))

def majorityDecoding(v,r):
    count1 = 0
    count0 = 0
    if r == 1:
        count1 += 1
    else:
        count0 += 1
    
    for i in v:
        if i == 1:
            count1 += 1
        else:
            count0 += 1
    
    if count1 > count0:
        return 1
    elif count1 < count0:
        return 0
    else:
        return int(np.random.randint(2,size = 1))

def majorityDecoding_c_hat_bec(P_c,L_c,r):
    if r != -1:
        return 0

    for i in P_c:
        if L_c[i] != -1:
            return 0
    else:
        return -1 

def majorityDecoding_bec(v,r):
    if r != -1:
        return 0

    for i in v:
        if i != -1:
            return 0
    else:
        return -1 

def get_likelihood(arr):
    lamda = (arr[0] / (arr[0] + 1)) * (1 - ((arr[1] / (arr[1] + 1)))) * (1 - ((arr[2] / (arr[2] + 1)))) + (arr[1] / (arr[1] + 1)) * (1 - ((arr[0] / (arr[0] + 1)))) * (1 - ((arr[2] / (arr[2] + 1)))) + (arr[2] / (arr[2] + 1)) * (1 - ((arr[0] / (arr[0] + 1)))) * (1 - ((arr[1] / (arr[1] + 1))))
    lamda += (arr[0] / (arr[0] + 1)) * (arr[1] / (arr[1] + 1)) * (arr[2] / (arr[2] + 1))
    
    if lamda != 1:
        return lamda / (1 - lamda)

def majorityDecoding_c_hat_SDD(P_c,L_c,r,p):
    lamda = 1
    if r == 1:
        lamda *= (1 - p) / p
    else:
        lamda *=  p / (1 - p)

    for i in P_c:
        lamda *= L_c[i]
    
    if lamda >= 1:
        return 1
    else:
        return 0

def majorityDecoding_SDD(v,r,p):
    lamda = 1
    if r == 1:
        lamda *= (1 - p) / p
    else:
        lamda *= p / (1 - p)

    for i in v:
        lamda *= i
    
    return lamda

def get_likelihood_bec(v):
    p_allZero = 1
    for i in v:
        p_allZero *= (1 - (i / (i + 1)))
    
    p_erasure = 1 - p_allZero
    if p_erasure != 1:
        lamda = p_erasure / (1 - p_erasure)
    else:
        lamda = 100000
    return lamda

def majorityDecoding_c_hat_SDD_bec(P_c,L_c,r,p):
    lamda = 1
    if r == -1:
        lamda *= 1
    else:
        lamda *=  0

    for i in P_c:
        lamda *= L_c[i]

    if lamda >= 1:
        return -1
    else:
        return 0

def majorityDecoding_SDD_bec(v,r,p):
    lamda = 1
    if r == -1:
        lamda *= 1
    else:
        lamda *=  0

    for i in v:
        lamda *= i
    
    return lamda

def numberOfErrors(m,rprime):
    n_error = 0 
    for i in range(len(m)):
        if m[i] != rprime[i]:
            n_error += 1
    
    return n_error

