import numpy as np


vecs = np.arange(2048*100).reshape(2048,100)
rank_top_2 = np.array([[1, 5, 8], [2, 4, 6]])
extract = vecs[:, rank_top_2]

top_num = 100
extract = np.ones((2048, 100, 117))
weight = (np.arange(top_num, 0, -1)/top_num).reshape(1, top_num, 1)
query_new = extract * weight
query_new = query_new.sum(axis=1)

print('>> done...')
