#%%
import argparse
import os
import time
import pickle
import pdb

import numpy as np

import torch
from torch.utils.model_zoo import load_url
from torch.autograd import Variable
from torchvision import transforms

from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.datasets.datahelpers import cid2filename
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.download import download_train, download_test
from cirtorch.utils.whiten import whitenlearn, whitenapply, pcawhitenlearn
from cirtorch.utils.evaluate import compute_map_and_print
from cirtorch.utils.general import get_data_root, htime

# hxq added
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy import ndimage
import math
import csv
import random
import gc
import pandas as pd
from cirtorch.datasets.datahelpers import clear_no_exist
# import faiss
#%%
PRETRAINED = {
    'retrievalSfM120k-vgg16-gem'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth',
    'retrievalSfM120k-resnet101-gem'    : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
}
datasets_names = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k',
                  'google-landmarks-dataset-resize', 'google-landmarks-dataset', 'google-landmarks-dataset-v2']
# whitening_names = ['retrieval-SfM-30k', 'retrieval-SfM-120k']
#%%
dataset = 'google-landmarks-dataset-v2'

gpu_id = '3'

# network_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/R101_F_GL/google-landmarks-dataset-resize_resnet101_gem_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362/model_epoch100.pth.tar'
network_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/R101_O_GL_FC/google-landmarks-dataset-resize_resnet101_gem_whiten_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362/model_epoch114.pth.tar'
# network_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/R101_O_GL_FC_multigem/google-landmarks-dataset-resize_resnet101_multigem_whiten_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362/model_epoch100.pth.tar'
# network_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/R101_O_120k_FC_multigem/retrieval-SfM-120k_resnet101_multigem_whiten_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize650/model_epoch101.pth.tar'
# network_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/R101_FC_origW_Mg1.5_GL_362/google-landmarks-dataset-resize_resnet101_gem_whiten_contrastive_m1.50_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362/model_epoch50.pth.tar'
# network_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/R101_M2_FC_120kLW_Mg1.5_GL_362/google-landmarks-dataset-resize_resnet101_gem_whiten_contrastive_m1.50_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362/model_epoch73.pth.tar'

image_size = 1024

# multiscale = '[1]'
# multiscale = '[1, 1/2**(1/2), 1/2]'
multiscale = '[1, 1/2**(1/2), 2**(1/2)]'
# multiscale = '[1, 0.875, 0.75]'
# multiscale = '[256/1600*(2**(1/2)),256/1600,256/1600*(2**(1/2))*(1/2)]'
ms = list(eval(multiscale))

whitening = 'google-landmarks-dataset-v2'
# whitening = 'google-landmarks-dataset-resize'

#%%
# check if there are unknown datasets
if dataset not in datasets_names:
    raise ValueError('Unsupported or unknown dataset: {}!'.format(dataset))

# setting up the visible GPU
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

# loading network from path
print(">> Loading network:\n>>>> '{}'".format(network_path))
if network_path in PRETRAINED:
    # pretrained networks (downloaded automatically)
    state = load_url(PRETRAINED[network_path], model_dir=os.path.join(get_data_root(), 'networks'))
else:
    # fine-tuned network from path
    state = torch.load(network_path)
# parsing net params from meta
# architecture, pooling, mean, std required
# the rest has default values, in case that is doesnt exist
net_params = {}
net_params['architecture'] = state['meta']['architecture']
net_params['pooling'] = state['meta']['pooling']
net_params['local_whitening'] = state['meta'].get('local_whitening', False)
net_params['regional'] = state['meta'].get('regional', False)
net_params['whitening'] = state['meta'].get('whitening', False)
net_params['mean'] = state['meta']['mean']
net_params['std'] = state['meta']['std']
net_params['pretrained'] = False
net_params['multi_layer_cat'] = state['meta']['multi_layer_cat']
# load network
net = init_network(net_params)
net.load_state_dict(state['state_dict'])

# if whitening is precomputed
if 'Lw' in state['meta']:
    net.meta['Lw'] = state['meta']['Lw']

print(">>>> loaded network: ")
print(net.meta_repr())

print(">> image size: {}".format(image_size))
# setting up the multi-scale parameters
if len(ms)>1 and net.meta['pooling'] == 'gem' and not net.meta['regional'] and not net.meta['whitening']:
    msp = net.pool.p.item()
    print(">> Set-up multiscale:")
    print(">>>> ms: {}".format(ms))
    print(">>>> msp: {}".format(msp))
else:
    msp = 1
    print(">> Set-up multiscale:")
    print(">>>> ms: {}".format(ms))
    print(">>>> msp: {}".format(msp))

# moving network to gpu and eval mode
net.cuda()
net.eval()

# set up the transform
normalize = transforms.Normalize(
    mean=net.meta['mean'],
    std=net.meta['std']
)
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])


#%%
start = time.time()
print('>> {}: Extracting...'.format(dataset))
#%%
print('>> Prepare data information...')
if dataset == 'google-landmarks-dataset':
    pass
elif dataset == 'google-landmarks-dataset-resize':
    pass
elif dataset == 'google-landmarks-dataset-v2':
    train_file_path = os.path.join(get_data_root(), 'train.csv')
    train_img_path = os.path.join(get_data_root(), 'train')


#%%
# print('>> load train image path...')
# csvfile = open(train_file_path, 'r')
# csvreader = csv.reader(csvfile)
# images = []
# landmarks = []
# for i, line in enumerate(csvreader):
#     if i != 0:
#         images.append(os.path.join(train_img_path, '/'.join(list(line[0])[:3]), line[0]+'.jpg'))
#         landmarks.append(int(line[2]))
# landmarks = np.array(landmarks)
# csvfile.close()
# print('>>>> train image total {}'.format(i))


#%%
print('>> load train image path...')
df_train = pd.read_csv(train_file_path)


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
select_landmark_id = [202510]
query_idx = np.array([df_train[df_train['landmark_id']==ld_id].index.to_numpy()
                      for ld_id in select_landmark_id]
                     ).squeeze()
qvecs = vecs[:, query_idx]


#%%
# kNN search
print('>> {}: Evaluating...'.format(dataset))
import faiss  # place it in the file top will cause network load so slowly
res = faiss.StandardGpuResources()
kNN_on_gpu_id = 0
top_num = 1000
k = 1000
index_threshold = 1000000
qvecs_T = np.zeros((qvecs.shape[1], qvecs.shape[0])).astype('float32')
qvecs_T[:] = qvecs.T[:]

print('>> find nearest neighbour, k: {}, top_num: {}'.format(k, top_num))
dis = []
ranks = []
num_list = list(range(0, vecs.shape[1] - 1, index_threshold))
num_list.append(vecs.shape[1])

for i in range(len(num_list)-1):
    vecs_T = np.zeros((num_list[i+1]-num_list[i], vecs.shape[0])).astype('float32')
    vecs_T[:] = vecs.T[num_list[i]:num_list[i + 1]]
    index = faiss.IndexFlatL2(vecs.shape[0])
    gpu_index = faiss.index_cpu_to_gpu(res, kNN_on_gpu_id, index)
    # gpu_index = faiss.index_cpu_to_all_gpus(index)  # use all GPU
    gpu_index.add(vecs_T)
    dis_temp, ranks_temp = gpu_index.search(qvecs_T, k)
    ranks_temp += num_list[i]
    dis.append(dis_temp)
    ranks.append(ranks_temp)
    del vecs_T, index, gpu_index, dis_temp, ranks_temp
    gc.collect()
    print('\r>>>> kNN search {} nearest neighbour {}/{} done...'.format(k, i + 1, len(num_list)-1), end='')
# import pickle
# with open(os.path.join('/media/iap205/Data4T/Export_temp/landmarks_view/202510_knn_1000',
#                        'dis_ranks.pkl'),'wb') as f:
#     pickle.dump((dis, ranks), f)
dis = np.hstack(tuple(dis))
ranks = np.hstack(tuple(ranks))
x_temp = np.tile(np.arange(dis.shape[0]).reshape(dis.shape[0], 1), (1, dis.shape[1]))
ranks_top_k = ranks[x_temp, np.argsort(dis, axis=1)][:, :k]
ranks_top_k = ranks_top_k.T
np.save(os.path.join('/media/iap205/Data4T/Export_temp/landmarks_view/202510_knn_1000', 'ranks_top_k.npy'), ranks_top_k)

# del scores, ranks
del qvecs_T, dis, ranks
gc.collect()
# del qvecs
# gc.collect()
print('')


#%%
ranks_temp = np.load(os.path.join('/media/iap205/Data4T/Export_temp/landmarks_view/202510_knn_1000','ranks_top_k.npy'))

#%%
import shutil
output_path = '/media/iap205/Data4T/Export_temp/landmarks_view/202510_knn_1000'
for i in range(ranks_top_k.shape[1]):
    knn_imgs = df_train['id'][ranks_top_k[:, i]].to_numpy()
    if not os.path.isdir(os.path.join(output_path, df_train['id'][query_idx[i]])):
        os.mkdir(os.path.join(output_path, df_train['id'][query_idx[i]]))
    for j in range(len(knn_imgs)):
        shutil.copyfile(os.path.join(train_img_path, '/'.join(list(knn_imgs[j])[:3]), knn_imgs[j]+'.jpg'),
                        os.path.join(output_path, df_train['id'][query_idx[i]], 'rank{}_{}.jpg'.format(j, knn_imgs[j])))
    print('\r>> {}/{} images copy done...'.format(i+1, ranks.shape[1]), end='')


#%%
# match images and calculate inliers
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
from skimage import feature
from skimage import measure
from skimage import transform
import tensorflow as tf

from tensorflow.python.platform import app
from delf import feature_io

cmd_args = None

_DISTANCE_THRESHOLD = 0.8
_INLIER_THRESHOLD = 30

tf.logging.set_verbosity(tf.logging.INFO)

features_output_dir = '/media/iap205/Data4T/Export_temp/GLD-v2_DELF_features'

for i in range(ranks.shape[1]):
    # Read features.
    features_1_path = os.path.join(features_output_dir, df_train['id'][query_idx[i]]+'.delf')
    image_1_path = os.path.join(train_img_path, '/'.join(list(df_train['id'][query_idx[i]])[:3]),
                                df_train['id'][query_idx[i]] + '.jpg')
    locations_1, _, descriptors_1, _, _ = feature_io.ReadFromFile(features_1_path)
    num_features_1 = locations_1.shape[0]
    # tf.logging.info("Loaded qurey image {}'s %d features".format(i) % num_features_1)
    d1_tree = spatial.cKDTree(descriptors_1)
    img_1 = mpimg.imread(image_1_path)
    for j in range(ranks.shape[0]):
        features_2_path = os.path.join(features_output_dir, knn_imgs[j]+'.delf')
        image_2_path = os.path.join(train_img_path, '/'.join(list(knn_imgs[j])[:3]), knn_imgs[j]+'.jpg')
        locations_2, _, descriptors_2, _, _ = feature_io.ReadFromFile(features_2_path)
        num_features_2 = locations_2.shape[0]
        # tf.logging.info("Loaded image 2's %d features" % num_features_2)

        # Find nearest-neighbor matches using a KD tree.
        _, indices = d1_tree.query(
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

        # tf.logging.info('Found %d inliers' % sum(inliers))

        if sum(inliers) >= _INLIER_THRESHOLD:
            # Visualize correspondences, and save to file.
            _, ax = plt.subplots()
            img_2 = mpimg.imread(image_2_path)
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
            ax.set_title('DELF correspondences')
            plt.savefig(os.path.join(output_path, df_train['id'][query_idx[i]],
                                     'inliers{}_{}'.format(sum(inliers), knn_imgs[j])))