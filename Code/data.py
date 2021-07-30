from Functions import *
import scipy.io as sc
H1 = sc.loadmat("Hmatrix")['H']
H2 = sc.loadmat("Hmatrix2")['H']
pos_col = []
for i in range(len(H2[0])):
    pos_col.append(list(locationOfOnes_col(H2[:,i])))
pos_raw = []
for i in range(len(H2)):
    pos_raw.append(list(locationOfOnes_raw(H2[i])))

# sc.savemat("test2.mat",{"pos_col":pos_col,"pos_raw":pos_raw})
