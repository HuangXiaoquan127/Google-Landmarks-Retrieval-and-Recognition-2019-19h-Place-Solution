#%%
import pickle
import pandas as pd
import numpy as np

GLD2_path = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2/google-landmarks-dataset-v2.pkl'
GLD2_cln_path = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2/train_cleaned_rm.npz'
GLD2_cln_m2_path = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2/train_cleaned_m2_rm.npz'
GLD1_resize_path = '/home/iap205/Datasets/google-landmarks-dataset-resize/google-landmarks-dataset-resize.pkl'
SfM_120k_path = '/home/iap205/Datasets/retrieval-SfM/train/retrieval-SfM-120k/retrieval-SfM-120k.pkl'
path = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2/google-landmarks-dataset-v2.pkl'


#%%
with open(GLD2_path, 'rb') as f:
    GLD2 = pickle.load(f)
print('pairs: {}'.format(len(GLD2['train']['qidxs'])))

num = np.hstack((np.array(GLD2['train']['qidxs']),
                 np.array(GLD2['train']['pidxs'])))
num = pd.DataFrame(num, columns=['idx'])
print('unique num: {}'.format(num.nunique()))

landmarks = np.array(GLD2['train']['landmark_id'])[np.hstack((np.array(GLD2['train']['qidxs']),
                                                              np.array(GLD2['train']['pidxs'])))]
landmarks = pd.DataFrame(landmarks, columns=['ld_id'])
print('unique landmarks num: {}'.format(landmarks.nunique()))


#%%
with open(GLD1_resize_path,'rb') as f:
    GLD1_resize = pickle.load(f)
print('pairs: {}'.format(len(GLD1_resize['train']['qidxs'])))
num = np.hstack((np.array(GLD1_resize['train']['qidxs']),
                 np.array(GLD1_resize['train']['pidxs'])))
num = pd.DataFrame(num, columns=['idx'])
print('unique num: {}'.format(num.nunique()))

landmarks = np.array(GLD1_resize['train']['landmark_id'])[np.hstack((np.array(GLD1_resize['train']['qidxs']),
                                                                     np.array(GLD1_resize['train']['pidxs'])))]
landmarks = pd.DataFrame(landmarks, columns=['ld_id'])
print('unique landmarks num: {}'.format(landmarks.nunique()))


#%%
with open(SfM_120k_path,'rb') as f:
    SfM_120k = pickle.load(f)
cluster = pd.DataFrame(SfM_120k['train']['cluster'], columns=['cluster'])
cluster.nunique()


#%%
GLD2_cln = np.load(GLD2_cln_path)
id, ld_id, qp_pairs = GLD2_cln['id'], GLD2_cln['ld_id'], GLD2_cln['qp_pairs']
num = np.hstack((qp_pairs[:, 0], qp_pairs[:, 1]))
num = pd.DataFrame(num, columns=['idx'])
print('unique num: {}'.format(num.nunique()))

landmarks = ld_id[np.hstack((qp_pairs[:, 0], qp_pairs[:, 1]))]
landmarks = pd.DataFrame(landmarks, columns=['ld_id'])
print('unique landmarks num: {}'.format(landmarks.nunique()))


#%%
with open(GLD2_cln_m2_path,'rb') as f:
    GLD2_cln_m2 = pickle.load(f)
