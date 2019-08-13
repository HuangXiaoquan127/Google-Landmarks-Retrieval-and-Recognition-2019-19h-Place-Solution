import numpy as np
d = 2048                           # dimension
nb = 4132914                      # database size
nq = 120000                       # nb of queries
# nb = 11000                      # database size
# nq = 1200                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')
import faiss                   # make faiss available


# res = faiss.StandardGpuResources()  # use a single GPU
# # build a flat (CPU) index
# index_flat = faiss.IndexFlatL2(d)
# # make it into a gpu index
# gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
#
# gpu_index_flat.add(xb)         # add vectors to the index
# print(gpu_index_flat.ntotal)
#
# k = 100                          # we want to see 4 nearest neighbors
# D, I = gpu_index_flat.search(xq, k)  # actual search
# print(I[:5])                   # neighbors of the 5 first queries
# print(I[-5:])                  # neighbors of the 5 last queries


ngpus = faiss.get_num_gpus()

print("number of GPUs:", ngpus)

cpu_index = faiss.IndexFlatL2(d)

gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
    cpu_index
)

nb = 1132914                      # database size
nq = 120000                       # nb of queries
# nb = 11000                      # database size
# nq = 1200                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

gpu_index.add(xb)              # add vectors to the index
print(gpu_index.ntotal)

k = 10                          # we want to see 4 nearest neighbors
D, I = gpu_index.search(xq, k) # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries


