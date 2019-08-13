#%%
import numpy as np
from scipy import spatial
from skimage import feature
from skimage import measure
from skimage import transform
import tensorflow as tf
import pandas as pd
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2 as cv

from tensorflow.python.platform import app
from delf import feature_io
import time
from cirtorch.utils.general import htime


#%%
# train_file_path = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2/train.csv'
# df_train = pd.read_csv(train_file_path)

#%%
# # img_path = './cirtorch/for_experiment/test/DELF/data/features'
# img_path = 'data/features'
# files = os.listdir(img_path)
# files_path = [os.path.join(img_path, file_name) for file_name in files]

#%%
img_dir = '/media/iap205/Data4T/Export_temp/landmarks_view/202510'
features_dir = '/media/iap205/Data4T/Export_temp/GLD-v2_DELF_features'
output_dir = os.path.join(img_dir, 'matched')
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

files = [file for file in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, file))]
img_path = [os.path.join(img_dir, file_name) for file_name in files]
features_path = [os.path.join(features_dir, file_name.split('.jpg')[0]+'.delf.npz') for file_name in files]

#%%
# img_dir = '/media/iap205/Data4T/Export_temp/landmarks_view/202510_knn_1000/01d598aa2ee31917'
# features_dir = '/media/iap205/Data4T/Export_temp/GLD-v2_DELF_features'
# output_dir = os.path.join(img_dir, 'matched')
# if not os.path.isdir(output_dir):
#     os.mkdir(output_dir)
#
# files = [file for file in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, file))]
# img_path = [os.path.join(img_dir, file_name) for file_name in files]
# features_path = [os.path.join(features_dir, file_name.split('_')[1]+'.delf.npz') for file_name in files]


#%%
# use faiss
import faiss
res = faiss.StandardGpuResources()
kNN_on_gpu_id = 0
ransac_lib = 'cv2' # None, skimage or cv2
start = time.time()
_DISTANCE_THRESHOLD = 0.64
_RANSAC_ITERS = 1000
inliers_record_faiss = []
for i in range(len(features_path)):
    # Read features.
    # locations_1, _, descriptors_1, _, _ = feature_io.ReadFromFile(features_path[i])
    features_1 = np.load(features_path[i])
    locations_1, descriptors_1 = features_1['loc'], features_1['desc']
    num_features_1 = locations_1.shape[0]
    # tf.logging.info("Loaded image 1's %d features" % num_features_1)
    output_dir_sub = os.path.join(output_dir, '{}_iter{}_{}'.format(ransac_lib, _RANSAC_ITERS, files[i].split('.jpg')[0]))
    if not os.path.isdir(output_dir_sub):
        os.mkdir(output_dir_sub)
    for j in range(len(features_path)):
        if i == 0 and j == 1:
            start = time.time()
        # locations_2, _, descriptors_2, _, _ = feature_io.ReadFromFile(features_path[j])
        features_2 = np.load(features_path[j])
        locations_2, descriptors_2 = features_2['loc'], features_2['desc']
        num_features_2 = locations_2.shape[0]
        # tf.logging.info("Loaded image 2's %d features" % num_features_2)

        # Find nearest-neighbor matches using a KD tree.
        index = faiss.IndexFlatL2(descriptors_1.shape[1])
        # gpu_index = faiss.index_cpu_to_gpu(res, kNN_on_gpu_id, index)
        gpu_index = index
        gpu_index.add(descriptors_1.astype('float32'))
        dis_temp, idxs_temp = gpu_index.search(descriptors_2.astype('float32'), 1)
        idxs_temp[dis_temp > _DISTANCE_THRESHOLD] = len(descriptors_1)
        idxs_temp = idxs_temp.squeeze()

        # Select feature locations for putative matches.
        locations_2_to_use = np.array([
            locations_2[i,]
            for i in range(num_features_2)
            if idxs_temp[i] != num_features_1
        ])
        locations_1_to_use = np.array([
            locations_1[idxs_temp[i],]
            for i in range(num_features_2)
            if idxs_temp[i] != num_features_1
        ])

        # Perform geometric verification using RANSAC.
        if ransac_lib == None or ransac_lib == 'skimage':
            _, inliers = measure.ransac((locations_1_to_use, locations_2_to_use),
                                        transform.AffineTransform,
                                        min_samples=3,
                                        residual_threshold=20,
                                        max_trials=_RANSAC_ITERS)
        elif ransac_lib == 'cv2':
            try:
                T, inliers = cv.estimateAffine2D(locations_1_to_use,
                                                 locations_2_to_use,
                                                 ransacReprojThreshold=20,
                                                 maxIters=_RANSAC_ITERS)
            except:
                pass

        inlier_num = inliers.sum()
        inliers_record_faiss.append(inlier_num)

        # Visualize correspondences, and save to file.
        try:
            _, ax = plt.subplots()
            img_1 = mpimg.imread(img_path[i])
            img_2 = mpimg.imread(img_path[j])
            inlier_idxs = np.nonzero(inliers)[0]
            feature.plot_matches(
                ax,
                img_1,
                img_2,
                locations_1_to_use,
                locations_2_to_use,
                np.column_stack((inlier_idxs, inlier_idxs)),
                matches_color='b')
            ax.axis('off')
            ax.set_title('{} inlier vs {} '.format(inlier_num, files[j].split('.jpg')[0]))
            plt.savefig(os.path.join(output_dir_sub, 'vs_{}_inlier_{}'.format(files[j].split('.jpg')[0], inlier_num)))
            plt.close('all')
            print('\r>> {}/{} match done'.format(i*len(features_path)+j+1, (len(features_path))**2), end='')
        except:
            pass
inliers_record_faiss = np.array(inliers_record_faiss).reshape(len(features_path), len(features_path))
print('>> feature match elapsed time: {:.5f}'.format(time.time() - start))


#%%
'''
# use scipy.spatial.cKDTree
start = time.time()
_DISTANCE_THRESHOLD = 0.8
inliers_record_scipy = []
for i in range(len(features_path)):
    # Read features.
    locations_1, _, descriptors_1, _, _ = feature_io.ReadFromFile(features_path[i])
    num_features_1 = locations_1.shape[0]
    # tf.logging.info("Loaded image 1's %d features" % num_features_1)
    for j in range(len(features_path)):
        locations_2, _, descriptors_2, _, _ = feature_io.ReadFromFile(features_path[j])
        num_features_2 = locations_2.shape[0]
        # tf.logging.info("Loaded image 2's %d features" % num_features_2)

        # Find nearest-neighbor matches using a KD tree.
        d1_tree = spatial.cKDTree(descriptors_1)
        diss, indices = d1_tree.query(
            descriptors_2, distance_upper_bound=_DISTANCE_THRESHOLD)

        # Select feature locations for putative matches.
        locations_2_to_use = np.array([
            locations_2[i,]
            for i in range(num_features_2)
            if indices[i] != num_features_1
        ])
        locations_1_to_use = np.array([
            locations_1[indices[i],]
            for i in range(num_features_2)
            if indices[i] != num_features_1
        ])

        # Perform geometric verification using RANSAC.
        _, inliers = measure.ransac((locations_1_to_use, locations_2_to_use),
                                    transform.AffineTransform,
                                    min_samples=3,
                                    residual_threshold=20,
                                    max_trials=1000)
        inliers_record_scipy.append(sum(inliers))
inliers_record_scipy = np.array(inliers_record_scipy).reshape(len(features_path), len(features_path))
print('>> feature match elapsed time: {:.5f}'.format(time.time() - start))
'''