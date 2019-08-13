# %%
import numpy as np
import pandas as pd
import os
import cv2 as cv
import sys


#%%
output_dir = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2'
train_file_path = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2/train.csv'
df_train = pd.read_csv(train_file_path)


#%%
# remove test set landmark to generate final train set
# first, statistic the test set's landmark ids
import pickle
test_pkl_path = '/home/iap205/Datasets/retrieval-SfM/test/google-landmarks-dataset-v2-test/' \
                'gnd_google-landmarks-dataset-v2-test.pkl'
with open(test_pkl_path, 'rb') as f:
    test_pkl = pickle.load(f)
test_ld_id = []
for i, id in enumerate(test_pkl['qimlist']):
    ld_id_temp = int(df_train.iloc[df_train[df_train['id'] == id].index, 2].values.squeeze())
    if ld_id_temp not in test_ld_id:
        test_ld_id.append(ld_id_temp)
    print('\r>>search {}/{} have {} unique landmark id...'
          .format(i+1, len(test_pkl['qimlist']), len(test_ld_id)), end='')

#%%
# second, remove train set positive pair that belong to test set landmarks
train_cleaned_m2 = pd.read_csv(os.path.join(output_dir, 'train_cleaned_mthod2.csv'))
train_cln_m2_dic = {train_cleaned_m2['landmark_id'][0]: [train_cleaned_m2['id'][0]]}
for i in range(len(train_cleaned_m2['landmark_id'])-1):
    if train_cleaned_m2['landmark_id'][i] == train_cleaned_m2['landmark_id'][i+1]:
        train_cln_m2_dic[train_cleaned_m2['landmark_id'][i]].append(train_cleaned_m2['id'][i+1])
    else:
        train_cln_m2_dic[train_cleaned_m2['landmark_id'][i+1]] = [train_cleaned_m2['id'][i+1]]

#%%
train_cleaned = np.load(os.path.join(output_dir, 'train_cleaned_rm.npz'))
id, ld_id, qp_pairs = train_cleaned['id'], train_cleaned['ld_id'], train_cleaned['qp_pairs']
train_cln_m2_rm = []
cnt = 0

for i in range(len(qp_pairs)):
    if ld_id[qp_pairs[i][0]] in train_cln_m2_dic.keys():
        if (id[qp_pairs[i][0]] in train_cln_m2_dic[ld_id[qp_pairs[i][0]]]) and \
           (id[qp_pairs[i][1]] in train_cln_m2_dic[ld_id[qp_pairs[i][0]]]):
            train_cln_m2_rm.append(qp_pairs[i])
train_cln_m2_rm = np.array(train_cln_m2_rm)
print('>> total {} positive pairs'.format(len(train_cln_m2_rm)))
np.savez(os.path.join(output_dir, 'train_cleaned_m2_rm.npz'),
         id=id, ld_id=ld_id, qp_pairs=np.array(train_cln_m2_rm))
