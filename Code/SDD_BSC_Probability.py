import numpy as np
import scipy.io as sc
import matplotlib.pyplot as py
import timeit 
from Functions import *
start = timeit.default_timer()
mat = sc.loadmat("Hmatrix2")
H = mat['H']
H = np.array(H,dtype=float)
# H = np.array([[1,0,0,0,0,1,0,1,0,1,0,0],
#               [1,0,0,1,1,0,0,0,0,0,1,0],
#               [0,1,0,0,1,0,1,0,1,0,0,0],
#               [0,0,1,0,0,0,1,1,0,0,0,1],
#               [0,0,1,0,0,1,0,0,0,0,1,1],
#               [0,1,0,0,1,0,0,0,1,0,1,0],
#               [1,0,0,1,0,0,1,0,0,1,0,0],
#               [0,1,0,0,0,1,0,1,0,1,0,0],
#               [0,0,1,1,0,0,0,0,1,0,0,1]], dtype =float)

L = (H.copy())
m = np.array(np.zeros((1,H.shape[1]),int)).flatten()
pos_col = sc.loadmat("test2")['pos_col']
# for i in range(len(m)):
#     pos_col.append(list(locationOfOnes_col(H[:,i])))

pos_raw = sc.loadmat("test2")['pos_raw']
# for i in range(len(H)):
#     pos_raw.append(list(locationOfOnes_raw(H[i])))

parr = np.arange(0.01,0.2,0.01)
pdecode = []
Nsim = 100
for p in parr:
    print(p)
    F_n = 0
    for itr in range(Nsim):
        print(itr , end = " ")
        r = bsc(m,p)
        #First iteration from VN to CN
        #instead of transmitting the probability we will transfer the likelihood ratio.
        for i in range(len(m)):
            for j in pos_col[i]:
                if(r[i] == 1):
                    L[j][i] = (1-p) / p #Likelihood ratio for transmitting 1 
                else:
                    L[j][i] = p / (1 - p)

        Lprime = L.copy()
        #First iteration from CN to VN
        for i in range(len(H)):
            for j in pos_raw[i]:
                arr = []
                for k in pos_raw[i]:
                    if k != j:
                        arr.append(Lprime[i][k])
                L[i][j] = get_likelihood(arr)

        #c hat in SDD after first iteration
        rprime = []
        for i in range(len(m)):
            rprime.append(majorityDecoding_c_hat_SDD(pos_col[i],L[:,i],r[i],p))

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
                    L[j][i] = majorityDecoding_SDD(arr,r[i],p)

            Lprime = L.copy()
            #t_th iteration from CN to VN
            for i in range(len(H)):
                for j in pos_raw[i]:
                    arr = []
                    for k in pos_raw[i]:
                        if k != j:
                            arr.append(Lprime[i][k])
                    L[i][j] = get_likelihood(arr)
            #DONE DONE DONE-----------------------------------------DONE DONE DONE DONE
            rprime = []
            for i in range(len(m)):
                rprime.append(majorityDecoding_c_hat_SDD(pos_col[i],L[:,i],r[i],p))
            if list(m) == rprime:
                F_n += 1
                break
            elif rprimeprev == rprime:
                break
    pdecode.append(F_n / Nsim)
    print()
    

stop = timeit.default_timer()
print(stop - start)
py.plot(parr,pdecode)
py.xlabel("p")
py.ylabel("Probability Of Decoding")
py.title(f"Probability Of Decoding vs p for Hmatrix2 (Nsim = {Nsim})")
py.grid()
py.show()