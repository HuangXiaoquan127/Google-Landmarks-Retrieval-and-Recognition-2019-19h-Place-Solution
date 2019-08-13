#%%
import pandas as pd
import os

#%%
# dir_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/train_cleaned_test/R101_GLD2_cleaned_362/GLD-v2-cleaned_resnet101_gem_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362'
dir_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/train_cleaned_test/R101_120k_362/retrieval-SfM-120k_resnet101_gem_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362/'
# dir_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/train_cleaned_test/R101_GLD1_resize_362/google-landmarks-dataset-resize_resnet101_gem_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362'
# dir_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/train_cleaned_test/R101_GLD2_362/google-landmarks-dataset-v2_resnet101_gem_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362'
# dir_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/train_cleaned_test/R101_GLD2_cleaned_m2_362/GLD-v2-cleaned-m2_resnet101_gem_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362'

csv_list = [path for path in os.listdir(dir_path) if path.split('.')[-1] == 'csv']
output_path = os.path.join(dir_path, csv_list[0].split('.csv')[0]+'_merge.csv')

#%%
# pd_temp = pd.read_csv(os.path.join(dir_path, csv_list[0]))
# pd_temp['epochs'] = list(range(1, len(pd_temp)+1))
# pd_temp.to_csv(os.path.join(dir_path, csv_list[0]), header=True, index=False)


#%%
for i in range(1, len(csv_list)):
    pd_temp = pd.read_csv(os.path.join(dir_path, csv_list[i]))
    if i == 1:
        pd_temp.to_csv(output_path, header=True, index=False)
    else:
        pd_temp.to_csv(output_path, header=False, index=False, mode='a')


#%%
pd_orig = pd.read_csv(os.path.join(dir_path, csv_list[0]))
temp_list = [1, ]
pd_temp = list()
for i in temp_list:
    pd_temp.append(pd.read_csv(os.path.join(dir_path, csv_list[i])))
pd_orig.to_csv(output_path, header=True, index=False)
#%%
for i in range(len(pd_temp)):
    for j in range(len(pd_temp[i])):
        if pd_temp[i]['epochs'][j] > pd_orig['epochs'][len(pd_orig)-1]:
            pd_temp[i].iloc[[j], :].to_csv(output_path, header=False, index=False, mode='a')
