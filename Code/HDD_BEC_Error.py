import numpy as np
import scipy.io as sc
import matplotlib.pyplot as py
import timeit 
from Functions import *
start = timeit.default_timer()
# mat = sc.loadmat("Hmatrix2")
# H = mat['H']
# H = np.array(H,dtype=int)
H = np.array([[1,0,0,0,0,1,0,1,0,1,0,0],
              [1,0,0,1,1,0,0,0,0,0,1,0],
              [0,1,0,0,1,0,1,0,1,0,0,0],
              [0,0,1,0,0,0,1,1,0,0,0,1],
              [0,0,1,0,0,1,0,0,0,0,1,1],
              [0,1,0,0,1,0,0,0,1,0,1,0],
              [1,0,0,1,0,0,1,0,0,1,0,0],
              [0,1,0,0,0,1,0,1,0,1,0,0],
              [0,0,1,1,0,0,0,0,1,0,0,1]])
L = H.copy()
m = np.array(np.zeros((1,H.shape[1]),int)).flatten()
pos_col = []#sc.loadmat("test2")['pos_col']
for i in range(len(m)):
    pos_col.append(list(locationOfOnes_col(H[:,i])))

pos_raw = []#sc.loadmat("test2")['pos_raw']
for i in range(len(H)):
    pos_raw.append(list(locationOfOnes_raw(H[i])))

parr = [0.1,0.2,0.3,0.4,0.5]
Nsim = 1000
perror = []
for p in parr:
    print(p)
    temp = []
    for itr in range(Nsim):
        print(itr,end = " ")
        n_error = []    
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

stop = timeit.default_timer()
print(stop - start)
trial = range(1,51)
perror = np.array(perror)
py.plot(trial,perror.transpose(),label = [f"p = {parr[0]}",f"p = {parr[1]}",f"p = {parr[2]}",f"p = {parr[3]}",f"p = {parr[4]}"])
py.xlabel("Number of iteration")
py.ylabel("Number of Errors")
py.title(f"Error vs Iteration for 9 x 12 H (Nsim = {Nsim})")
py.grid()
py.legend()
py.show()

