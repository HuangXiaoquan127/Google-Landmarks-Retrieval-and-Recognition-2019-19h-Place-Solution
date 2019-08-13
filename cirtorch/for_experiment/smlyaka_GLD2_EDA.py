#%%
import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt

#%%
dir_path = '/media/iap205/Data4T/Datasets/cleaned_subsets_train2019'
csv_list = os.listdir(dir_path)

#%%
pd_vf = {}
for i, name in enumerate(csv_list):
    name_temp = name.split('train19_cleaned_verifythresh')[-1].split('_freqthresh')
    name_temp[-1] = name_temp[-1].split('.csv')[0]
    name_temp = f'v{name_temp[0]}f{name_temp[1]}'
    pd_vf[name_temp] = pd.read_csv(os.path.join(dir_path, csv_list[i]))

#%%
pd_vf['v20f1'].info()
#%%
pd_vf['v20f1'].head()


#%%
pd_vf['v30f2'].info()

#%%
pd_vf['v30f2'][pd_vf['v30f2']['landmark_id']==202510].index


#%%
vf_list = list(pd_vf.keys())
for i, vf in enumerate(vf_list):
    output_dir = f'/media/iap205/Data4T/Export_temp/landmarks_view/smlyaka_GLD2_cln_verify/{vf}'
    img_dir = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2/train'
    idx = pd_vf[vf][pd_vf[vf]['landmark_id'] == 202510].index.values
    img_temp = list(pd_vf[vf].iloc[idx, :]['images'].values[0].split(' '))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if not os.path.isdir(os.path.join(output_dir, '202510')):
        os.mkdir(os.path.join(output_dir, '202510'))

    for img in img_temp:
        shutil.copyfile(os.path.join(img_dir, '/'.join(list(img)[:3]), img+'.jpg'),
                    os.path.join(output_dir, '202510', img+'.jpg'))
    print(f'>> {vf} copy done...')





