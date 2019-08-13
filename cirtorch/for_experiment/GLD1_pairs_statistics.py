#%%
import pickle
import numpy as np
import os
import pandas as pd


#%%
pkl_path = '/home/iap205/Datasets/google-landmarks-dataset-resize/google-landmarks-dataset-resize.pkl'

with open(pkl_path, 'rb') as f:
    train = pickle.load(f)

#%%
# id len: 1197322
# qidxs len: 313946
# print('id len: {}'.format(len(train['train']['id'])))
print('pairs num: {}'.format(len(train['train']['qidxs'])))

# %%
use_idxs = np.concatenate((train['train']['qidxs'], train['train']['pidxs']))
print('pair use id num: {}/{}'.format(pd.DataFrame(use_idxs).nunique().values[0], len(train['train']['id'])))
