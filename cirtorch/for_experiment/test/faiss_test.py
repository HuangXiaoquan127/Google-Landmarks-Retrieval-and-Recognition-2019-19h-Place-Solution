import numpy as np
d = 2048                           # dimension
nb = 1093278                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
# xb = np.random.random((nb, d)).astype('float32')
# xb = np.zeros((nb, d)).astype('float32')
# xb[:, 0] += np.arange(nb) / 1000.
# xq = np.random.random((nq, d)).astype('float32')
# xq[:, 0] += np.arange(nq) / 1000.

vecs = np.zeros((2048, 1093))
qvecs = np.zeros((2048, 115)).astype('float32')
qvecs2 = np.zeros((115, 2048)).astype('float32')

import faiss                   # make faiss available
# index = faiss.IndexFlatL2(xb.shape[1])   # build the index
# print(index.is_trained)
# index.add(xb)                  # add vectors to the index
# print(index.ntotal)

index = faiss.IndexFlatL2(2048)   # build the index
print(index.is_trained)
# vecs_T = np.zeros((vecs.shape[1], vecs.shape[0])).astype('float32')
# vecs_T[:] = vecs.T[:]
# index.add(qvecs2.T.astype('float32'))                  # add vectors to the index
c = qvecs.reshape(-1,1).reshape()
del qvecs
index.add(c)
print(index.ntotal)

k = 4                          # we want to see 4 nearest neighbors
D, I = index.search(qvecs2, k) # sanity check
print(I)
print(D)
D, I = index.search(xq, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries

