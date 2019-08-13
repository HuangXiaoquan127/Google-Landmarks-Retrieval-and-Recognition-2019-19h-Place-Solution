import numpy as np
import os
import pickle
import h5py
import pandas as pd

#%%
print('>>>> load train vecs from pkl...')
vecs_pkl_path = '/media/iap205/Data4T/Export_temp/google-landmarks-dataset-v2-vecs-qvecs/' \
                'R101_FC_GL_256/GL2_train_1024_MS1, 0.707, 1.414'
split_num = int(6 * 4)
vecs = []
for i in range(split_num):
    # vecs_temp = np.loadtxt(open(os.path.join(get_data_root(), 'index_vecs{}_of_{}.csv'.format(i+1, split_num)), "rb"),
    #                        delimiter=",", skiprows=0)
    with open(os.path.join(vecs_pkl_path, 'train_vecs{}_of_{}.pkl'.format(i+1, split_num)), 'rb') as f:
        vecs.append(pickle.load(f))
    print('\r>>>> train_vecs{}_of_{}.pkl load done...'.format(i+1, split_num), end='')
vecs = np.hstack(tuple(vecs))
print('')

#%%
output_path = '/media/iap205/Data4T/Export_temp/google-landmarks-dataset-v2-vecs-qvecs/' \
              'R101_FC_GL_256/GL2_train_1024_MS1, 0.707, 1.414/train_vecs_all.npy'
np.save(output_path, vecs)



#%%
input_path = '/media/iap205/Data4T/Export_temp/google-landmarks-dataset-v2-vecs-qvecs/' \
              'R101_FC_GL_256/GL2_train_1024_MS1, 0.707, 1.414/train_vecs_all.npy'
vecs = np.load(input_path)

#%%
vecs_T = np.zeros((vecs.shape[1], vecs.shape[0])).astype('float32')
vecs_T[:] = vecs.T[:]

#%%
output_path = '/media/iap205/Data4T/Export_temp/google-landmarks-dataset-v2-vecs-qvecs/' \
              'R101_FC_GL_256/GL2_train_1024_MS1, 0.707, 1.414/train_vecs_all_T.npy'
np.save(output_path, vecs_T)



#%%
train_file_path = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2/train.csv'
df_train = pd.read_csv(train_file_path)

#%%
output_path = '/media/iap205/Data4T/Export_temp/google-landmarks-dataset-v2-vecs-qvecs/' \
              'R101_FC_GL_256/GL2_train_1024_MS1, 0.707, 1.414/train_vecs_all.h5'
f = h5py.File(output_path, 'w')
f['ids_train'] = df_train['id'].to_numpy().astype('S16')
f['feats_train'] = vecs.T
f.close()



