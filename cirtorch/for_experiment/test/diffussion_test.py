from cirtorch.utils.diffussion import *
import numpy as np


Q = np.random.random((2048, 118000))
X = np.random.random((2048, 1100000))

K = 73 # approx 50 mutual nns
QUERYKNN = 66
alpha = 0.9

# search, rank, and print
# sim  = np.dot(X.T, Q)
sim = np.random.random((1100000, 118000))
qsim = sim_kernel(sim).T

sortidxs = np.argsort(-qsim, axis = 1)
for i in range(len(qsim)):
    qsim[i,sortidxs[i,QUERYKNN:]] = 0

qsim = sim_kernel(qsim)
A = np.dot(X.T, X)
W = sim_kernel(A).T
W = topK_W(W, K)
Wn = normalize_connection_graph(W)

cg_ranks =  cg_diffusion(qsim, Wn, alpha)