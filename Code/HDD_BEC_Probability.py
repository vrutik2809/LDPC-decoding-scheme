import numpy as np
import scipy.io as sc
import matplotlib.pyplot as py
import timeit 
from Functions import *
start = timeit.default_timer()
mat = sc.loadmat("Hmatrix")
H = mat['H']
H = np.array(H,dtype= int)
# H = np.array([[1,0,0,0,0,1,0,1,0,1,0,0],
#               [1,0,0,1,1,0,0,0,0,0,1,0],
#               [0,1,0,0,1,0,1,0,1,0,0,0],
#               [0,0,1,0,0,0,1,1,0,0,0,1],
#               [0,0,1,0,0,1,0,0,0,0,1,1],
#               [0,1,0,0,1,0,0,0,1,0,1,0],
#               [1,0,0,1,0,0,1,0,0,1,0,0],
#               [0,1,0,0,0,1,0,1,0,1,0,0],
#               [0,0,1,1,0,0,0,0,1,0,0,1]])
L = H.copy()
m = np.array(np.zeros((1,H.shape[1]),int)).flatten()
pos_col = sc.loadmat("test")['pos_col']
# for i in range(len(m)):
#     pos_col.append(list(locationOfOnes_col(H[:,i])))

pos_raw = sc.loadmat("test")['pos_raw']
# for i in range(len(H)):
#     pos_raw.append(list(locationOfOnes_raw(H[i])))

parr = np.arange(0.0001,1,0.1)
pdecode = []
Nsim = 100
for p in parr:
    print(p)
    F_n = 0
    for itr in range(Nsim):
        print(itr,end = " ")  
        r = bec(m,p)
        L = H.copy()

        #First iteration from VN to CN
        for i in range(len(m)):
            for j in pos_col[i]:
                L[j][i] = (r[i])

        Lprime = L.copy()
        #First iteration from CN to VN
        for i in range(len(H)):
            for j in pos_raw[i]:
                arr = []
                for k in pos_raw[i]:
                    if k != j:
                        arr.append(Lprime[i][k])
                #transmitting logic
                for l in arr:
                    if l == -1:
                        L[i][j] = -1
                        break
                else:
                    L[i][j] = sum(arr) % 2

        #After First iteration c_hat 
        rprime = []
        for i in range(len(m)):
            rprime.append(majorityDecoding_c_hat_bec(pos_col[i],L[:,i],r[i]))

        for t in range(2,51):
            rprimeprev = rprime
            Lprime = L.copy()
            #t_th iteration from VN to CN
            for i in range(len(m)):
                for j in pos_col[i]:
                    arr = []
                    for k in pos_col[i]:
                        if k != j:
                            arr.append(Lprime[k][i])
                    L[j][i] = majorityDecoding_bec(arr,r[i])

            Lprime = L.copy()
            #t_th iteration from CN to VN
            for i in range(len(H)):
                for j in pos_raw[i]:
                    arr = []
                    for k in pos_raw[i]:
                        if k != j:
                            arr.append(Lprime[i][k])
                    
                    #transmitting logic
                    for l in arr:
                        if l == -1:
                            L[i][j] = -1
                            break
                    else:
                        L[i][j] = sum(arr) % 2

            #After t_th iteration c_hat
            rprime = []
            for i in range(len(m)):
                rprime.append(majorityDecoding_c_hat_bec(pos_col[i],L[:,i],r[i]))

            if list(m) == rprime:
                F_n += 1
                break
            elif rprimeprev == rprime:
                break
    print()
    pdecode.append(F_n / Nsim)

stop = timeit.default_timer()
print(stop - start)
py.plot(parr,pdecode)
py.xlabel("p")
py.ylabel("Probability Of Decoding")
py.title(f"Probability Of Decoding vs p for Hmatrix2 (Nsim = {Nsim})")
py.grid()
py.show()

