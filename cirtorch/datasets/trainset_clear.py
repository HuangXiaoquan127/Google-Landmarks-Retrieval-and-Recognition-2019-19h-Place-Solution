# %%
import numpy as np
import pandas as pd
import os
import cv2 as cv
import sys
import time
from cirtorch.utils.general import htime

part = 1
gpu_id = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
np.random.seed(0)

# %%
print('>> load train.csv...')
train_file_path = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2/train.csv'
df_train = pd.read_csv(train_file_path)
df_train_sort = df_train.sort_values('landmark_id')

split_num = 6
landmarks_fold = pd.DataFrame(df_train_sort['landmark_id'].value_counts())
landmarks_fold.reset_index(inplace=True)
landmarks_fold.columns = ['landmark_id', 'count']
landmarks_fold = landmarks_fold.sort_values('landmark_id')
part_num = int(len(landmarks_fold) / split_num)
ld_num_list = list(range(0, len(landmarks_fold) + 1, part_num))
ld_num_list[-1] = len(landmarks_fold)
num_list = [0]
for i in range(1, len(ld_num_list)):
    num_list.append(landmarks_fold['count'][:ld_num_list[i]].sum())

# %%
print('>> generate train part dictionary...')
part = part
features_dir = '/media/iap205/Data4T/Export_temp/GLD-v2_DELF_features'
output_dir = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2'
train_part = df_train_sort[num_list[part]:num_list[part + 1]].reset_index()
# train_part = df_train_sort[landmarks_fold[:ld_num_list[part]+23548].sum()['count']
#                            :landmarks_fold[:ld_num_list[part]+23548].sum()['count'] + 31].reset_index()
train_part_dic = {train_part['landmark_id'][0]: [train_part['id'][0]]}
for i in range(len(train_part) - 1):
    if train_part['landmark_id'][i] == train_part['landmark_id'][i + 1]:
        train_part_dic[train_part['landmark_id'][i]].append(train_part['id'][i + 1])
    else:
        train_part_dic[train_part['landmark_id'][i + 1]] = [train_part['id'][i + 1]]


# %%
import faiss
import h5py

res = faiss.StandardGpuResources()
# The CUDA environment variables have been set before, so choose the GPUs in them here.
kNN_on_gpu_id = 0
_DISTANCE_THRESHOLD = 0.64
_RANSAC_ITERS = 1000
_INLIERS_THRESHOLD = 30
_NUM_PER_LD_THRESHOLD = 50
qidxs, pidxs = [], []
# orig_qidxs, orig_pidxs = [], []
idxs_cnt = 0

# Also generate another cleanup method table
_FREQUENCY_THRESHOLD = 2
id_cleaned, ld_id_cleaned = [], []
verified_cnt = 0
added_flag = False

print('>> start cleaning...')
for i, ld_id in enumerate(train_part_dic):
    features_path = [os.path.join(features_dir, file_name + '.delf.npz')
                     for file_name in train_part_dic[ld_id][:min(len(train_part_dic[ld_id]), _NUM_PER_LD_THRESHOLD)]]

    for j in range(len(features_path)):
        features_1 = np.load(features_path[j])
        locations_1, descriptors_1 = features_1['loc'], features_1['desc']
        if descriptors_1.size == 0:
            continue
        num_features_1 = locations_1.shape[0]

        for k in range(len(features_path)):
            if train_part_dic[ld_id][j] != train_part_dic[ld_id][k]:
                features_2 = np.load(features_path[k])
                locations_2, descriptors_2 = features_2['loc'], features_2['desc']
                # There may be a feature not found, the array is [] empty, which will cause subsequent knn search errors
                if descriptors_2.size == 0:
                    continue
                num_features_2 = locations_2.shape[0]
                index = faiss.IndexFlatL2(descriptors_1.shape[1])
                gpu_index = faiss.index_cpu_to_gpu(res, kNN_on_gpu_id, index)
                # gpu_index = index
                gpu_index.add(descriptors_1.astype('float32'))
                dis_temp, idxs_temp = gpu_index.search(descriptors_2.astype('float32'), 1)
                idxs_temp[dis_temp > _DISTANCE_THRESHOLD] = len(descriptors_1)
                idxs_temp = idxs_temp[:, 0]

                locations_2_to_use = np.array([locations_2[i, :] for i in range(num_features_2)
                                               if idxs_temp[i] != num_features_1])
                locations_1_to_use = np.array([locations_1[idxs_temp[i], :] for i in range(num_features_2)
                                               if idxs_temp[i] != num_features_1])

                # Because there may be no matching points found
                if locations_2_to_use.size != 0:
                    _, inliers = cv.estimateAffine2D(locations_1_to_use, locations_2_to_use,
                                                     ransacReprojThreshold=20, maxIters=_RANSAC_ITERS)
                else:
                    inliers = np.array([])

                if inliers.sum() > _INLIERS_THRESHOLD:
                    qidxs.append(idxs_cnt + j)
                    pidxs.append(idxs_cnt + k)
                    if not added_flag:
                        verified_cnt += 1
                        if verified_cnt > _FREQUENCY_THRESHOLD:
                            added_flag = True
                            id_cleaned.append(train_part_dic[ld_id][j])
                            ld_id_cleaned.append(ld_id)
        verified_cnt = 0
        added_flag = False
    idxs_cnt += len(train_part_dic[ld_id])
    print('\r>> {}/{} landmark clear up...'.format(i + 1, len(train_part_dic)), end='')

np.savez(os.path.join(output_dir, 'train_cleaned_part{}_of_{}.npz'.format(part + 1, split_num)),
         id=train_part['id'].to_numpy(), ld_id=train_part['landmark_id'].to_numpy(),
         qp_pairs=np.hstack((np.array(qidxs).reshape(-1, 1), np.array(pidxs).reshape(-1, 1))))
np.savez(os.path.join(output_dir, 'train_cleaned_part{}_of_{}_method2.npz'.format(part + 1, split_num)),
         id=np.array(id_cleaned), ld_id=np.array(ld_id_cleaned))

# %%
# merge all parts
id, ld_id, qp_pairs = [], [], []
method2_id, method2_ld_id = [], []
for i in range(split_num):
    train_cleaned = np.load(os.path.join(output_dir, 'train_cleaned_part{}_of_{}.npz'
                                         .format(i + 1, split_num)))
    id.append(train_cleaned['id'])
    ld_id.append(train_cleaned['ld_id'])
    qp_pairs_temp = train_cleaned['qp_pairs'] + num_list[i]
    qp_pairs.append(qp_pairs_temp)

    train_cleaned_m2 = np.load(os.path.join(output_dir, 'train_cleaned_part{}_of_{}_method2.npz'
                                            .format(i + 1, split_num)))
    method2_id.append(train_cleaned_m2['id'])
    method2_ld_id.append(train_cleaned_m2['ld_id'])

id = np.hstack(tuple(id))
ld_id = np.hstack(tuple(ld_id))
qp_pairs = np.vstack(tuple(qp_pairs))
np.savez(os.path.join(output_dir, 'train_cleaned.npz'), id=id, ld_id=ld_id, qp_pairs=qp_pairs)

method2_id = np.hstack(tuple(method2_id))
method2_ld_id = np.hstack(tuple(method2_ld_id))
pd.DataFrame(np.vstack((method2_id, method2_ld_id)).T, columns=['id', 'landmark_id'])\
             .to_csv(os.path.join(output_dir, 'train_cleaned_mthod2.csv'), index=False)


#%%
# verified the cleaned result
import shutil
np.random.seed(0)
verified_output = '/media/iap205/Data4T/Export_temp/landmarks_view/cleaned_verified'
train_img_path = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2/train'
train_cleaned = np.load(os.path.join(output_dir, 'train_cleaned.npz'))
id, ld_id, qp_pairs = train_cleaned['id'], train_cleaned['ld_id'], train_cleaned['qp_pairs']
sample = np.random.permutation(len(train_cleaned['qp_pairs']))[:2000]
qp = ['q', 'p']
for i, idx in enumerate(sample):
    if ld_id[qp_pairs[idx][0]] != ld_id[qp_pairs[idx][1]]:
        print('>>The cleaning result is wrong!')
    # verified_output_path = os.path.join(verified_output, 'pair_{}'.format(i))
    for j in range(len(qp_pairs[idx])):
        orig_img_path = os.path.join(train_img_path, '/'.join(list(id[qp_pairs[idx][j]])[:3]), id[qp_pairs[idx][j]]+'.jpg')
        shutil.copyfile(orig_img_path,
                        os.path.join(verified_output, 'pair_{}_{}_{}.jpg'.format(idx, qp[j], id[qp_pairs[idx][j]])))
    print('\r>> cope {}/{} done...'.format(i, len(sample)), end='')


#%%
# pairs exploration data analysis to verify the threshold of pairs num of per landmark
import matplotlib.pyplot as plt
import seaborn as sns

train_cleaned = np.load(os.path.join(output_dir, 'train_cleaned.npz'))
id, ld_id, qp_pairs = train_cleaned['id'], train_cleaned['ld_id'], train_cleaned['qp_pairs']
pairs_per_ld_id = np.zeros((len(landmarks_fold), 1), dtype='int')
for idx in qp_pairs:
    pairs_per_ld_id[ld_id[idx]] += 1
pairs_per_ld_id = pd.DataFrame(np.hstack((landmarks_fold['landmark_id'].to_numpy().reshape(-1, 1), pairs_per_ld_id)),
                               columns=['ld_id', 'pairs_cnt']).to_numpy()

sns.set()
per_ld_id_count_fold = pd.DataFrame(pairs_per_ld_id['pairs_cnt'].value_counts())
per_ld_id_count_fold.reset_index(inplace=True)
per_ld_id_count_fold.columns = ['pairs_cnt', 'count']
ax = per_ld_id_count_fold['count'].plot(loglog=True, grid=True)
ax.grid(b=True, which='both')
locs, labels = plt.xticks()
plt.setp(labels, rotation=30)
ax.set(xlabel="Number of pairs", ylabel="Number of landmarks")
plt.savefig(os.path.join('/home/iap205/Pictures', 'per_ld_id_count_fold'), dpi='figure')
plt.show()

sns.set()
ax = pairs_per_ld_id.plot.scatter(\
     x='ld_id', y='pairs_cnt',
     title='Training set: number of images per class(statter plot)')
locs, labels = plt.xticks()
plt.setp(labels, rotation=30)
ax.set(xlabel="Landmarks", ylabel="Number of pairs")
plt.savefig(os.path.join('/home/iap205/Pictures', 'per_ld_id_count'), dpi='figure')
plt.show()

sns.set()
ax = pairs_per_ld_id.boxplot(column='pairs_cnt')
ax.grid(b=True, which='both')
ax.set_yscale('log')
plt.savefig(os.path.join('/home/iap205/Pictures', 'per_ld_id_count_box'), dpi='figure')
plt.show()


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
train_cleaned = np.load(os.path.join(output_dir, 'train_cleaned.npz'))
id, ld_id, qp_pairs = train_cleaned['id'], train_cleaned['ld_id'], train_cleaned['qp_pairs']
pairs_per_ld_id = np.zeros((len(landmarks_fold), 1), dtype='int')
for idx in qp_pairs:
    pairs_per_ld_id[ld_id[idx]] += 1
pairs_per_ld_id = pd.DataFrame(np.hstack((landmarks_fold['landmark_id'].to_numpy().reshape(-1, 1), pairs_per_ld_id)),
                               columns=['ld_id', 'pairs_cnt']).to_numpy()
qp_pairs_rm = []
pairs_index = 0
np.random.seed(0)
_PAIRS_PER_LD_THRESHOLD = 100
for ld_id_temp, pairs_cnt in pairs_per_ld_id:
    if ld_id_temp not in test_ld_id:
        if pairs_cnt > _PAIRS_PER_LD_THRESHOLD:
            qp_pairs_rm.append(qp_pairs[pairs_index + np.random.permutation(pairs_cnt)[:_PAIRS_PER_LD_THRESHOLD], :])
        else:
            qp_pairs_rm.append(qp_pairs[pairs_index:pairs_index+pairs_cnt, :])
    pairs_index += pairs_cnt
qp_pairs_rm = np.vstack(tuple(qp_pairs_rm))
print('>> total {} positive pairs'.format(len(qp_pairs_rm)))
np.savez(os.path.join(output_dir, 'train_cleaned_rm.npz'),
         id=id, ld_id=ld_id, qp_pairs=np.array(qp_pairs_rm))


# %%
# if __name__ == '__main__':
#     clear(int(sys.argv[1]), int(sys.argv[2]))

