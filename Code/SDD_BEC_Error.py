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

parr = [0.1,0.2,0.3,0.4,0.5]
Nsim = 1
perror = []
for p in parr:
    print(p)
    temp = []
    for itr in range(Nsim):
        # print(itr, end = " ")
        r = bec(m,p)
        # print(list(r).count(-1))
        n_error = []
        #First iteration from VN to CN
        #instead of transmitting the probability we will transfer the likelihood ratio.
        for i in range(len(m)):
            for j in pos_col[i]:
                if(r[i] == -1):
                    L[j][i] = 1  # lamda for transfering erasure = P(e | r) / p(0 | r)
                else:
                    L[j][i] = 0

        Lprime = L.copy()
        #First iteration from CN to VN
        for i in range(len(H)):
            for j in pos_raw[i]:
                arr = []
                for k in pos_raw[i]:
                    if k != j:
                        arr.append(Lprime[i][k])
                L[i][j] = get_likelihood_bec(arr)

        #c hat in SDD after first iteration
        rprime = []
        for i in range(len(m)):
            rprime.append(majorityDecoding_c_hat_SDD_bec(pos_col[i],L[:,i],r[i],p))
        n_error.append(rprime.count(-1))

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
                    L[j][i] = majorityDecoding_SDD_bec(arr,r[i],p)

            Lprime = L.copy()
            #t_th iteration from CN to VN
            for i in range(len(H)):
                for j in pos_raw[i]:
                    arr = []
                    for k in pos_raw[i]:
                        if k != j:
                            arr.append(Lprime[i][k])
                    L[i][j] = get_likelihood_bec(arr)
            #DONE DONE DONE-----------------------------------------DONE DONE DONE DONE
            rprime = []
            for i in range(len(m)):
                rprime.append(majorityDecoding_c_hat_SDD_bec(pos_col[i],L[:,i],r[i],p))
            n_error.append(rprime.count(-1))
            # if rprimeprev == rprime:
            #     break
        temp.append(n_error)
    print()
    flag = []
    temp = np.array(temp)
    for s in range(50):
        flag.append(sum(temp[:,s]) / Nsim)
    perror.append(flag)

# print(perror.count(-1))
# print(list(perror))    
stop = timeit.default_timer()
print(stop-start)
trial = range(1,51)
perror = np.array(perror)
py.plot(trial,perror.transpose(),label = [f"p = {parr[0]}",f"p = {parr[1]}",f"p = {parr[2]}",f"p = {parr[3]}",f"p = {parr[4]}"])
py.xlabel("Number of iteration")
py.ylabel("Number of Errors")
py.title(f"Error vs Iteration for Hmatrix2 (Nsim = {Nsim})")
py.grid()
py.legend()
py.show()