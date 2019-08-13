# %%
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
from cirtorch.datasets.landmarks_downloader import ParseData
import csv
from cirtorch.utils.diffussion import *

# %%
PRETRAINED = {
    'retrievalSfM120k-vgg16-gem': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth',
    'retrievalSfM120k-resnet101-gem': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
}

datasets_names = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k', 'google-landmarks-dataset-resize-test',
                  'google-landmarks-dataset-v2-test']
whitening_names = ['retrieval-SfM-30k', 'retrieval-SfM-120k']

# %%
test_datasets = 'google-landmarks-dataset-v2-test'

gpu_id = '3'

# network_path = '/media/iap205/Data/Export/cnnimageretrieval-pytorch/trained_network/retrieval-SfM-120k_resnet101_gem_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362/model_epoch98.pth.tar'
# network_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/R101_F_GL/google-landmarks-dataset-resize_resnet101_gem_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362/model_epoch100.pth.tar'
# network_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/R101_O_GL_FC/google-landmarks-dataset-resize_resnet101_gem_whiten_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362/model_epoch114.pth.tar'
network_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/R101_O_GL_FC_multigem/google-landmarks-dataset-resize_resnet101_multigem_whiten_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362/model_epoch100.pth.tar'
# network_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/R101_O_120k_FC_multigem/retrieval-SfM-120k_resnet101_multigem_whiten_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize650/model_epoch101.pth.tar'

params = [
    {'ms': [1], 'image_size': [256, 362, 512, 768, 1024, 1280, float('inf')], 'pac_dims': [2048, 1536, 1024, 512]},
    {'ms': [1, 1 / 2 ** (1 / 2), 1 / 2], 'image_size': [1024], 'pac_dims': [2048, 1536, 1024, 512]},
    {'ms': [1, 1 / 2 ** (1 / 2), 2 ** (1 / 2)], 'image_size': [1024],
     'pac_dims': [2048, 1536, 1024, 512, 128, 96, 64, 32]}]

whitening = 'google-landmarks-dataset-v2-test'

# %%
# check if there are unknown datasets
for dataset in test_datasets.split(','):
    if dataset not in datasets_names:
        raise ValueError('Unsupported or unknown dataset: {}!'.format(dataset))
# setting up the visible GPU
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
# loading network from path
if network_path is not None:

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

# %%
start = time.time()
print('>> {}: Extracting...'.format(dataset))

# prepare config structure for the test dataset
cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))
images = [cfg['im_fname'](cfg, i) for i in range(cfg['n'])]
qimages = [cfg['qim_fname'](cfg, i) for i in range(cfg['nq'])]
# bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]
print('>> not use bbxs...')
bbxs = None

# %%
for param in params:
    for image_size in param['image_size']:
        # setting up the multi-scale parameters
        print(">> image size: {}".format(image_size))
        if len(param['ms']) > 1 and net.meta['pooling'] == 'gem' \
                and not net.meta['regional'] and not net.meta['whitening']:
            msp = net.pool.p.item()
            print(">> Set-up multiscale:")
            print(">>>> ms: {}".format(param['ms']))
            print(">>>> msp: {}".format(msp))
        else:
            msp = 1
            print(">> Set-up multiscale:")
            print(">>>> ms: {}".format(param['ms']))
            print(">>>> msp: {}".format(msp))

        # extract database and query vectors
        print('>> {}: database images...'.format(dataset))
        vecs = extract_vectors(net, images, image_size, transform, ms=param['ms'], msp=msp)
        print('>> {}: query images...'.format(dataset))
        qvecs = extract_vectors(net, qimages, image_size, transform, bbxs=bbxs, ms=param['ms'], msp=msp)
        print('>> {}: Evaluating...'.format(dataset))
        # convert to numpy
        vecs = vecs.numpy()
        qvecs = qvecs.numpy()
        # search, rank, and print
        scores = np.dot(vecs.T, qvecs)
        ranks = np.argsort(-scores, axis=0)
        qvecs_orig = qvecs
        ranks_orig = ranks
        mismatched_info = compute_map_and_print(dataset, ranks, cfg['gnd'], kappas=[1, 5, 10, 100])

        # compute whitening
        if whitening is not None:
            start = time.time()
            if 'Lw' in net.meta and whitening in net.meta['Lw']:
                print('>> {}: Whitening is precomputed, loading it...'.format(whitening))
                if len(param['ms']) > 1:
                    Lw = net.meta['Lw'][whitening]['ms']
                else:
                    Lw = net.meta['Lw'][whitening]['ss']
            else:
                # if we evaluate networks from path we should save/load whitening
                # not to compute it every time
                if network_path is not None:
                    whiten_fn = network_path + '_{}_whiten'.format(whitening)
                    if len(param['ms']) > 1:
                        whiten_fn += '_ms'
                    whiten_fn += '.pth'
                else:
                    whiten_fn = None
                if whiten_fn is not None and os.path.isfile(whiten_fn):
                    print('>> {}: Whitening is precomputed, loading it...'.format(whitening))
                    Lw = torch.load(whiten_fn)
                else:
                    print('>> {}: Learning whitening...'.format(whitening))
                    # extract whitening vectors
                    print('>> {}: Extracting...'.format(whitening))
                    # wvecs = vecs
                    wvecs = np.hstack((vecs, qvecs))
                    # learning whitening
                    print('>> {}: Learning...'.format(whitening))
                    m, P = pcawhitenlearn(wvecs)
                    # m, P = whitenlearn(wvecs)
                    Lw = {'m': m, 'P': P}
                    # saving whitening if whiten_fn exists
                    if whiten_fn is not None:
                        print('>> {}: Saving to {}...'.format(whitening, whiten_fn))
                        # torch.save(Lw, whiten_fn)
            print('>> {}: elapsed time: {}'.format(whitening, htime(time.time() - start)))
        else:
            Lw = None

        if Lw is not None:
            for dim in param['pac_dims']:
                print('>>>> pac_dim: {}'.format(dim))
                # whiten the vectors
                vecs_lw = whitenapply(vecs, Lw['m'], Lw['P'], dim)
                qvecs_lw = whitenapply(qvecs, Lw['m'], Lw['P'], dim)

                # search, rank, and print
                scores = np.dot(vecs_lw.T, qvecs_lw)
                ranks_lw = np.argsort(-scores, axis=0)
                qvecs_lw_orig = qvecs_lw
                ranks_lw_orig = ranks_lw
                mismatched_info = compute_map_and_print(dataset + ' + whiten', ranks_lw, cfg['gnd'],
                                                        kappas=[1, 5, 10, 100])

# %%
# save vecs, qvecs, ranks to pickle
import pickle

file_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/test_vecs/R101_O_GL_FC_multigem/vecs_qvecs.pkl'
with open(file_path, 'wb') as f:
    pickle.dump((vecs, qvecs, ranks), f)
# %%
# load vecs, qvecs, ranks from pickle
import pickle

file_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/test_vecs/R101_O_GL_FC_multigem/vecs_qvecs.pkl'
with open(file_path, 'rb') as f:
    (vecs, qvecs, ranks) = pickle.load(f)
qvecs_orig = qvecs
ranks_orig = ranks
mismatched_info = compute_map_and_print(dataset, ranks, cfg['gnd'], kappas=[1, 5, 10, 100])

# save vecs_lw, qvecs_lw, ranks_lw to pickle
import pickle

file_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/test_vecs/R101_O_GL_FC_multigem/vecs_lw_qvecs_lw.pkl'
with open(file_path, 'wb') as f:
    pickle.dump((vecs_lw, qvecs_lw, ranks_lw), f)
# %%
# load vecs, qvecs, ranks from pickle
import pickle

file_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/test_vecs/R101_O_GL_FC_multigem/vecs_lw_qvecs_lw.pkl'
with open(file_path, 'rb') as f:
    (vecs_lw, qvecs_lw, ranks_lw) = pickle.load(f)
qvecs_lw_orig = qvecs_lw
ranks_lw_orig = ranks_lw
mismatched_info = compute_map_and_print(dataset, ranks_lw, cfg['gnd'], kappas=[1, 5, 10, 100])


# %%
# find the best k for query expansion
ks = list(range(1, int(10317 / 491) + 1))
ranks_split_num = 1
for k in ks:
    QE_weight = (np.arange(k, 0, -1) / k).reshape(1, k, 1)
    print('>> query expansion top k: {}'.format(k))
    for i in range(ranks_split_num):
        ranks_split = ranks[:k, int(ranks.shape[1] / ranks_split_num * i):
                                int(ranks.shape[1] / ranks_split_num * (i + 1))]
        top_k_vecs = vecs[:, ranks_split]  # shape = (2048, k, query_split_size)
        qvecs_temp = (top_k_vecs * QE_weight).sum(axis=1)
        qvecs_temp = qvecs_temp / (np.linalg.norm(qvecs_temp, ord=2, axis=0, keepdims=True) + 1e-6)
        if i == 0:
            qvecs = qvecs_temp
        else:
            qvecs = np.hstack((qvecs, qvecs_temp))
        print('\r>>>> calculate new query vectors {}/{} done...'.format(i + 1, ranks_split_num), end='')
    print('')
    # search, rank, and print
    scores = np.dot(vecs.T, qvecs)
    ranks = np.argsort(-scores, axis=0)
    mismatched_info = compute_map_and_print(dataset, ranks, cfg['gnd'], kappas=[1, 5, 10, 100])
    # recover the original value
    qvecs = qvecs_orig
    ranks = ranks_orig


# %%
# Query expansion
k = 8
alpha = 8. / 2
iters = 6
ranks_split_num = 1
for iter in range(iters):
    print('>> Query expansion: k: {}, alpha: {}, iteration: {}'.format(k, alpha, iter + 1))
    QE_weight = (np.arange(k, 0, -1) / k).reshape(1, k, 1)
    for i in range(ranks_split_num):
        ranks_split = ranks[:k, int(ranks.shape[1] / ranks_split_num * i):
                                int(ranks.shape[1] / ranks_split_num * (i + 1))]
        top_k_vecs = vecs[:, ranks_split]  # shape = (2048, k, query_split_size)
        qvecs_temp = (top_k_vecs * (QE_weight ** alpha)).sum(axis=1)
        # qvecs_temp += qvecs
        qvecs_temp = qvecs_temp / (np.linalg.norm(qvecs_temp, ord=2, axis=0, keepdims=True) + 1e-6)
        if i == 0:
            qvecs = qvecs_temp
        else:
            qvecs = np.hstack((qvecs, qvecs_temp))
        print('\r>>>> calculate new query vectors {}/{} done...'.format(i + 1, ranks_split_num), end='')
    print('')
    # k += 1
    # search, rank, and print
    scores = np.dot(vecs.T, qvecs)
    ranks = np.argsort(-scores, axis=0)
    mismatched_info = compute_map_and_print(dataset, ranks, cfg['gnd'], kappas=[1, 5, 10, 100])
# recover the original value
qvecs = qvecs_orig
ranks = ranks_orig


# %%
# looking for the best pca and query expansion combination parameters
for dim in [2048]:
    print('>>>> pac_dim: {}'.format(dim))
    # whiten the vectors
    vecs_lw = whitenapply(vecs, Lw['m'], Lw['P'], dim)
    qvecs_lw = whitenapply(qvecs, Lw['m'], Lw['P'], dim)

    # search, rank, and print
    scores = np.dot(vecs_lw.T, qvecs_lw)
    ranks_lw = np.argsort(-scores, axis=0)
    qvecs_lw_orig = qvecs_lw
    ranks_lw_orig = ranks_lw
    mismatched_info = compute_map_and_print(dataset + ' + whiten', ranks_lw, cfg['gnd'], kappas=[1, 5, 10, 100])

    ks = [15]
    alphas = [12. / 2]
    iters = 6
    ranks_split_num = 1
    for k in ks:
        for alpha in alphas:
            for iter in range(iters):
                print('>> Query expansion: k: {}, alpha: {}, iteration: {}'.format(k, alpha, iter + 1))
                QE_weight = (np.arange(k, 0, -1) / k).reshape(1, k, 1)
                for i in range(ranks_split_num):
                    ranks_split = ranks_lw[:k, int(ranks_lw.shape[1] / ranks_split_num * i):
                                               int(ranks_lw.shape[1] / ranks_split_num * (i + 1))]
                    top_k_vecs = vecs[:, ranks_split]  # shape = (2048, k, query_split_size)
                    qvecs_temp = (top_k_vecs * (QE_weight ** alpha)).sum(axis=1)
                    # qvecs_temp += qvecs
                    qvecs_temp = qvecs_temp / (np.linalg.norm(qvecs_temp, ord=2, axis=0, keepdims=True) + 1e-6)
                    if i == 0:
                        qvecs_lw = qvecs_temp
                    else:
                        qvecs_lw = np.hstack((qvecs_lw, qvecs_temp))
                    print('\r>>>> calculate new query vectors {}/{} done...'.format(i + 1, ranks_split_num), end='')
                print('')
                # k += 1
                # search, rank, and print
                scores = np.dot(vecs.T, qvecs_lw)
                ranks_lw = np.argsort(-scores, axis=0)
                mismatched_info = compute_map_and_print(dataset, ranks_lw, cfg['gnd'], kappas=[1, 5, 10, 100])
            # recover the original value
            qvecs_lw = qvecs_lw_orig
            ranks_lw = ranks_lw_orig
            print('')


# %%
# diffussion
vecs_lw = whitenapply(vecs, Lw['m'], Lw['P'], dim)
qvecs_lw = whitenapply(qvecs, Lw['m'], Lw['P'], dim)
# qvecs_lw = qvecs_lw_orig
# ranks_lw = ranks_lw_orig
Q = qvecs_lw
X = vecs_lw
K = 100  # approx 50 mutual nns
QUERYKNN = 10
R = 2000
alpha = 0.9

# search, rank, and print
sim = np.dot(X.T, Q)
qsim = sim_kernel(sim).T

sortidxs = np.argsort(-qsim, axis=1)
for i in range(len(qsim)):
    qsim[i, sortidxs[i, QUERYKNN:]] = 0

qsim = sim_kernel(qsim)
A = np.dot(X.T, X)
W = sim_kernel(A).T
W = topK_W(W, K)
Wn = normalize_connection_graph(W)

plain_ranks = np.argsort(-sim, axis=0)
mismatched_info = compute_map_and_print(dataset + ' + whiten', plain_ranks, cfg['gnd'], kappas=[1, 5, 10, 100])

cg_ranks = cg_diffusion(qsim, Wn, alpha)
mismatched_info = compute_map_and_print(dataset + ' + whiten', cg_ranks, cfg['gnd'], kappas=[1, 5, 10, 100])

cg_trunk_ranks = dfs_trunk(sim, A, alpha=alpha, QUERYKNN=QUERYKNN)
mismatched_info = compute_map_and_print(dataset + ' + whiten', cg_trunk_ranks, cfg['gnd'], kappas=[1, 5, 10, 100])

fast_spectral_ranks = fsr_rankR(qsim, Wn, alpha, R)
mismatched_info = compute_map_and_print(dataset + ' + whiten', fast_spectral_ranks, cfg['gnd'], kappas=[1, 5, 10, 100])


