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
import csv
import random
import gc
from cirtorch.datasets.datahelpers import clear_no_exist

# import faiss
# %%
PRETRAINED = {
    'retrievalSfM120k-vgg16-gem': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth',
    'retrievalSfM120k-resnet101-gem': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
}
datasets_names = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k',
                  'google-landmarks-dataset-resize', 'google-landmarks-dataset', 'google-landmarks-dataset-v2']
# whitening_names = ['retrieval-SfM-30k', 'retrieval-SfM-120k']
# %%
dataset = 'google-landmarks-dataset-v2'

gpu_id = '3'

network_path = None
# network_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/R101_F_GL/google-landmarks-dataset-resize_resnet101_gem_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362/model_epoch100.pth.tar'
# network_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/R101_O_GL_FC/google-landmarks-dataset-resize_resnet101_gem_whiten_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362/model_epoch114.pth.tar'
# network_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/R101_O_GL_FC_multigem/google-landmarks-dataset-resize_resnet101_multigem_whiten_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362/model_epoch100.pth.tar'
# network_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/R101_O_120k_FC_multigem/retrieval-SfM-120k_resnet101_multigem_whiten_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize650/model_epoch101.pth.tar'
# network_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/R101_FC_origW_Mg1.5_GL_362/google-landmarks-dataset-resize_resnet101_gem_whiten_contrastive_m1.50_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362/model_epoch50.pth.tar'

network_offtheshelf = None
network_offtheshelf = 'resnet101-gem'
# network_offtheshelf = 'resnext101_32x8d-gem'
# network_offtheshelf = 'resnet101-learnpool'

image_size = 1024

multiscale = '[1]'
# multiscale = '[1, 1/2**(1/2), 1/2]'
# multiscale = '[1, 1/2**(1/2), 2**(1/2)]'
# multiscale = '[1, 0.875, 0.75]'
# multiscale = '[256/1600*(2**(1/2)),256/1600,256/1600*(2**(1/2))*(1/2)]'
ms = list(eval(multiscale))

whitening = 'google-landmarks-dataset-v2'
whitening = 'retrieval-SfM-120k'
# whitening = 'google-landmarks-dataset-resize'

multi_layer_cat = 1

# %%
# check if there are unknown datasets
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

# loading offtheshelf network
elif network_offtheshelf is not None:
    # parse off-the-shelf parameters
    offtheshelf = network_offtheshelf.split('-')
    net_params = {}
    net_params['architecture'] = offtheshelf[0]
    net_params['pooling'] = offtheshelf[1]
    net_params['local_whitening'] = 'lwhiten' in offtheshelf[2:]
    net_params['regional'] = 'reg' in offtheshelf[2:]
    net_params['whitening'] = 'whiten' in offtheshelf[2:]
    net_params['pretrained'] = True
    net_params['multi_layer_cat'] = multi_layer_cat

    # load off-the-shelf network
    print(">> Loading off-the-shelf network:\n>>>> '{}'".format(network_offtheshelf))
    net = init_network(net_params)
    print(">>>> loaded network: ")
    print(net.meta_repr())

print(">> image size: {}".format(image_size))
# setting up the multi-scale parameters
if len(ms) > 1 and net.meta['pooling'] == 'gem' and not net.meta['regional'] and not net.meta['whitening']:
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
# loading db
db_root = os.path.join(get_data_root(), 'train', whitening)
ims_root = os.path.join(db_root, 'ims')
db_fn = os.path.join(db_root, '{}-whiten.pkl'.format(whitening))
with open(db_fn, 'rb') as f:
    db = pickle.load(f)
images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]


# %%
# extract whitening vectors
print('>> {}: Extracting...'.format(whitening))
split_num = 4
extract_num = int(len(images) / split_num)
num_list = list(range(0, len(images) + 1, extract_num))
num_list[-1] = len(images)
# %%
part = [0]
for k in part:
    print('>>>> extract part {} of {}'.format(k + 1, split_num))
    wvecs = extract_vectors(net, images[num_list[k]:num_list[k + 1]], image_size, transform, ms=ms, msp=msp)
    wvecs = wvecs.numpy()
    print('>>>> save whitening vecs to pkl...')
    wvecs_file_path = os.path.join(get_data_root(), 'whitening_vecs{}_of_{}.pkl'.format(k + 1, split_num))
    wvecs_file = open(wvecs_file_path, 'wb')
    pickle.dump(wvecs, wvecs_file)
    wvecs_file.close()
    print('>>>> whitening_vecs{}_of_{}.pkl save done...'.format(k + 1, split_num))


# %%
print('>>>> load whitening vecs from pkl...')
split_num = 4
for i in range(split_num):
    with open(os.path.join(get_data_root(), 'whitening_vecs{}_of_{}.pkl'.format(i + 1, split_num)), 'rb') as f:
        wvecs_temp = pickle.load(f)
    if i == 0:
        wvecs = wvecs_temp
    else:
        wvecs = np.hstack((wvecs, wvecs_temp[:, :]))
    del wvecs_temp
    gc.collect()
    print('\r>>>> whitening_vecs{}_of_{}.pkl load done...'.format(i + 1, split_num), end='')
print('')


#%%
# learning whitening
print('>> {}: Learning...'.format(whitening))
m, P = whitenlearn(wvecs, db['qidxs'], db['pidxs'])
Lw = {'m': m, 'P': P}

# saving whitening if whiten_fn exists
whiten_fn = os.path.join(get_data_root(), 'whiten',
                         'R101_M2_120k_IS1024_MS1_WL.pth')
if whiten_fn is not None:
    print('>> {}: Saving to {}...'.format(whitening, whiten_fn))
    torch.save(Lw, whiten_fn)
