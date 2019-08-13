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
network_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/R101_M2_FC_120kLW_Mg1.5_GL_362/google-landmarks-dataset-resize_resnet101_gem_whiten_contrastive_m1.50_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362/model_epoch73.pth.tar'

network_offtheshelf = None
# network_offtheshelf = 'resnet101-gem'

image_size = 1024

# multiscale = '[1]'
# multiscale = '[1, 1/2**(1/2), 1/2]'
multiscale = '[1, 1/2**(1/2), 2**(1/2)]'
# multiscale = '[1, 0.875, 0.75]'
# multiscale = '[256/1600*(2**(1/2)),256/1600,256/1600*(2**(1/2))*(1/2)]'
ms = list(eval(multiscale))

whitening = 'google-landmarks-dataset-v2'
# whitening = 'google-landmarks-dataset-resize'

multi_layer_cat = 2

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

# %%
start = time.time()
print('>> {}: Extracting...'.format(dataset))


# %%
# for google-landmarks-dataset and google-landmarks-dataset-resize
print('>> Prepare data information...')
if dataset == 'google-landmarks-dataset':
    index_file_path = os.path.join(get_data_root(), 'index.csv')
    index_mark_path = os.path.join(get_data_root(), 'index_mark.csv')
    index_miss_path = os.path.join(get_data_root(), 'index_miss.csv')
    index_img_path = os.path.join(get_data_root(), 'index')
    test_file_path = os.path.join(get_data_root(), 'test.csv')
    test_mark_path = os.path.join(get_data_root(), 'test_mark.csv')
    test_mark_add_path = os.path.join(get_data_root(), 'test_mark_add.csv')
    test_miss_path = os.path.join(get_data_root(), 'test_miss.csv')
    test_img_path = os.path.join(get_data_root(), 'google-landmarks-dataset-test')
elif dataset == 'google-landmarks-dataset-resize':
    index_file_path = os.path.join(get_data_root(), 'index.csv')
    index_mark_path = os.path.join(get_data_root(), 'resize_index_mark.csv')
    index_miss_path = os.path.join(get_data_root(), 'resize_index_miss.csv')
    index_img_path = os.path.join(get_data_root(), 'resize_index_image')
    test_file_path = os.path.join(get_data_root(), 'test.csv')
    test_mark_path = os.path.join(get_data_root(), 'resize_test_mark.csv')
    test_mark_add_path = os.path.join(get_data_root(), 'resize_test_mark_add.csv')
    test_miss_path = os.path.join(get_data_root(), 'resize_test_miss.csv')
    test_img_path = os.path.join(get_data_root(), 'resize_test_image')
elif dataset == 'google-landmarks-dataset-v2':
    pass
    test_img_path = os.path.join(get_data_root(), 'test-v2')
if not (os.path.isfile(index_mark_path) or os.path.isfile(index_miss_path)):
    clear_no_exist(index_file_path, index_mark_path, index_miss_path, index_img_path)
if not (os.path.isfile(test_mark_path) or os.path.isfile(test_miss_path)):
    clear_no_exist(test_file_path, test_mark_path, test_miss_path, test_img_path)
# %%
print('>> load index image path...')
retrieval_other_dataset = '/home/iap205/Datasets/google-landmarks-dataset-resize'
csvfile = open(index_mark_path, 'r')
csvreader = csv.reader(csvfile)
images = []
miss, add = 0, 0
for line in csvreader:
    if line[0] == '1':
        images.append(os.path.join(index_img_path, line[1] + '.jpg'))
    elif line[0] == '0':
        retrieval_img_path = os.path.join(retrieval_other_dataset, 'resize_index_image', line[1] + '.jpg')
        if os.path.isfile(retrieval_img_path):
            images.append(retrieval_img_path)
            add += 1
        miss += 1
csvfile.close()
print('>>>> index image miss: {}, supplement: {}, still miss: {}'.format(miss, add, miss - add))
# %%
print('>> load query image path...')
csvfile = open(test_mark_path, 'r')
csvreader = csv.reader(csvfile)
savefile = open(test_mark_add_path, 'w')
save_writer = csv.writer(savefile)
qimages = []
miss, add = 0, 0
for line in csvreader:
    if line[0] == '1':
        qimages.append(os.path.join(test_img_path, line[1] + '.jpg'))
        save_writer.writerow(line)
    elif line[0] == '0':
        retrieval_img_path = os.path.join(retrieval_other_dataset, 'resize_test_image', line[1] + '.jpg')
        if os.path.isfile(retrieval_img_path):
            qimages.append(retrieval_img_path)
            save_writer.writerow(['1', line[1]])
            add += 1
        else:
            save_writer.writerow(line)
        miss += 1
csvfile.close()
savefile.close()
print('>>>> test image miss: {}, supplement: {}, still miss: {}'.format(miss, add, miss - add))



# %%
# for google-landmarks-dataset-v2
print('>> Prepare data information...')
index_file_path = os.path.join(get_data_root(), 'index-v2.csv')
index_img_path = os.path.join(get_data_root(), 'index-v2')
test_file_path = os.path.join(get_data_root(), 'test-v2.csv')
test_mark_add_path = os.path.join(get_data_root(), 'test-v2_mark_add.csv')
test_img_path = os.path.join(get_data_root(), 'test-v2')

print('>> load index image path...')
csvfile = open(index_file_path, 'r')
csvreader = csv.reader(csvfile)
images = []
for i, line in enumerate(csvreader):
    if i != 0:
        images.append(os.path.join(index_img_path, '/'.join(list(line[0])[:3]), line[0] + '.jpg'))
csvfile.close()
print('>>>> index image total: {}'.format(len(images)))
# %%
print('>> load query image path...')
csvfile = open(test_file_path, 'r')
csvreader = csv.reader(csvfile)
savefile = open(test_mark_add_path, 'w')
save_writer = csv.writer(savefile)
qimages = []
miss, add = 0, 0
for i, line in enumerate(csvreader):
    if i != 0:
        qimages.append(os.path.join(test_img_path, '/'.join(list(line[0])[:3]), line[0] + '.jpg'))
        save_writer.writerow(['1', line[0]])
csvfile.close()
savefile.close()
print('>>>> test image total: {}'.format(len(qimages)))


# %%
# extract index vectors
print('>> {}: database images...'.format(dataset))
split_num = 8
extract_num = int(len(images) / split_num)
num_list = list(range(0, len(images) + 1, extract_num))
num_list[-1] = len(images)
# %%
part = [0, 1]
for k in part:
    print('>>>> extract part {} of {}'.format(k + 1, split_num))
    vecs = extract_vectors(net, images[num_list[k]:num_list[k + 1]], image_size, transform, ms=ms, msp=msp)
    vecs = vecs.numpy()
    print('>>>> save index vecs to pkl...')
    vecs_file_path = os.path.join(get_data_root(), 'index_vecs{}_of_{}.pkl'.format(k + 1, split_num))
    vecs_file = open(vecs_file_path, 'wb')
    pickle.dump(vecs, vecs_file)
    vecs_file.close()
    print('>>>> index_vecs{}_of_{}.pkl save done...'.format(k + 1, split_num))

# %%
# extract query vectors
print('>> {}: query images...'.format(dataset))
split_num = 8
extract_num = int(len(qimages) / split_num)
num_list = list(range(0, len(qimages) + 1, extract_num))
num_list[-1] = len(qimages)
# %%
part = [0, 1]
for k in part:
    print('>>>> extract part {} of {}'.format(k + 1, split_num))
    qvecs = extract_vectors(net, qimages[num_list[k]:num_list[k + 1]], image_size, transform, ms=ms, msp=msp)
    qvecs = qvecs.numpy()
    print('>>>> save test vecs to pkl...')
    qvecs_file_path = os.path.join(get_data_root(), 'test_vecs{}_of_{}.pkl'.format(k + 1, split_num))
    qvecs_file = open(qvecs_file_path, 'wb')
    pickle.dump(qvecs, qvecs_file)
    qvecs_file.close()
    print('>>>> test_vecs{}_of_{}.pkl save done...'.format(k + 1, split_num))


# %%
print('>>>> load index vecs from pkl...')
split_num = 8
for i in range(split_num):
    # vecs_temp = np.loadtxt(open(os.path.join(get_data_root(), 'index_vecs{}_of_{}.csv'.format(i+1, split_num)), "rb"),
    #                        delimiter=",", skiprows=0)
    with open(os.path.join(get_data_root(), 'index_vecs{}_of_{}.pkl'.format(i + 1, split_num)), 'rb') as f:
        vecs_temp = pickle.load(f)
    if i == 0:
        vecs = vecs_temp
    else:
        vecs = np.hstack((vecs, vecs_temp[:, :]))
    del vecs_temp
    gc.collect()
    print('\r>>>> index_vecs{}_of_{}.pkl load done...'.format(i + 1, split_num), end='')
print('')
# %%
print('>>>> load test vecs from pkl...')
split_num = 8
for i in range(split_num):
    # qvecs_temp = np.loadtxt(open(os.path.join(get_data_root(), 'test_vecs{}_of_{}.csv'.format(i+1, split_num)), "rb"),
    #                         delimiter=",", skiprows=0)
    with open(os.path.join(get_data_root(), 'test_vecs{}_of_{}.pkl'.format(i + 1, split_num)), 'rb') as f:
        qvecs_temp = pickle.load(f)
    if i == 0:
        qvecs = qvecs_temp
    else:
        qvecs = np.hstack((qvecs, qvecs_temp[:, :]))
    del qvecs_temp
    gc.collect()
    print('\r>>>> test_vecs{}_of_{}.pkl load done...'.format(i + 1, split_num), end='')
print('')


# %%
# compute whitening
if whitening is not None:
    start = time.time()
    if 'Lw' in net.meta and whitening in net.meta['Lw']:
        print('>> {}: Whitening is precomputed, loading it...'.format(whitening))
        if len(ms) > 1:
            Lw = net.meta['Lw'][whitening]['ms']
        else:
            Lw = net.meta['Lw'][whitening]['ss']
    else:
        # if we evaluate networks from path we should save/load whitening
        # not to compute it every time
        if network_path is not None:
            whiten_fn = network_path + '_{}_whiten'.format(whitening)
            if len(ms) > 1:
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
            Lw = {'m': m, 'P': P}
            del wvecs
            gc.collect()
            # saving whitening if whiten_fn exists
            if whiten_fn is not None:
                whiten_fn = os.path.join(get_data_root(), 'whiten',
                                         'R101_FC_origW_Mg1.5_GL_362_IS1024_MS1,0.707,1.414.pth')
                print('>> {}: Saving to {}...'.format(whitening, whiten_fn))
                torch.save(Lw, whiten_fn)
    print('>> {}: elapsed time: {}'.format(whitening, htime(time.time() - start)))
else:
    Lw = None

# %%
print('>> apply PCA whiten...')
if Lw is not None:
    # whiten the vectors and shrinkage
    vecs_lw = np.dot(Lw['P'], vecs - Lw['m'])
    print('>> index vecs PCA whitening done...')
    qvecs_lw = np.dot(Lw['P'], qvecs - Lw['m'])
    print('>> test vecs PCA whitening done...')
# %%
print('>>>> save index PCA whiten vecs to pkl...')
split_num = 8
extract_num = int(len(images) / split_num)
num_list = list(range(0, len(images) + 1, extract_num))
num_list[-1] = len(images)
for i in range(split_num):
    vecs_file_path = os.path.join(get_data_root(), 'index_PCA_whiten_vecs{}_of_{}.pkl'.format(i + 1, split_num))
    vecs_file = open(vecs_file_path, 'wb')
    pickle.dump(vecs_lw[:, num_list[i]:num_list[i + 1]], vecs_file)
    vecs_file.close()
    print('\r>>>> index_PCA_whiten_vecs{}_of_{}.pkl save done...'.format(i + 1, split_num), end='')
print('')

print('>>>> save test PCA whiten vecs to pkl...')
split_num = 8
extract_num = int(len(qimages) / split_num)
num_list = list(range(0, len(qimages) + 1, extract_num))
num_list[-1] = len(qimages)
for i in range(split_num):
    qvecs_file_path = os.path.join(get_data_root(), 'test_PCA_whiten_vecs{}_of_{}.pkl'.format(i + 1, split_num))
    qvecs_file = open(qvecs_file_path, 'wb')
    pickle.dump(qvecs_lw[:, num_list[i]:num_list[i + 1]], qvecs_file)
    qvecs_file.close()
    print('\r>>>> test_PCA_whiten_vecs{}_of_{}.pkl save done...'.format(i + 1, split_num), end='')
print('')


# %%
split_num = 8
print('>>>> load index PCA whiten vecs from pkl...')
for i in range(split_num):
    with open(os.path.join(get_data_root(), 'index_PCA_whiten_vecs{}_of_{}.pkl'.format(i + 1, split_num)), 'rb') as f:
        vecs_temp = pickle.load(f)
    if i == 0:
        vecs_lw = vecs_temp
    else:
        vecs_lw = np.hstack((vecs_lw, vecs_temp[:, :]))
    del vecs_temp
    gc.collect()
    print('\r>>>> index_PCA_whiten_vecs{}_of_{}.pkl load done...'.format(i + 1, split_num), end='')
print('')

split_num = 8
print('>>>> load test PCA whiten vecs from pkl...')
for i in range(split_num):
    with open(os.path.join(get_data_root(), 'test_PCA_whiten_vecs{}_of_{}.pkl'.format(i + 1, split_num)), 'rb') as f:
        qvecs_temp = pickle.load(f)
    if i == 0:
        qvecs_lw = qvecs_temp
    else:
        qvecs_lw = np.hstack((qvecs_lw, qvecs_temp[:, :]))
    del qvecs_temp
    gc.collect()
    print('\r>>>> test_PCA_whiten_vecs{}_of_{}.pkl load done...'.format(i + 1, split_num), end='')
print('')

# %%
# extract principal components and normalization
vecs = vecs_lw
qvecs = qvecs_lw
ratio = 1
vecs = vecs[:int(vecs.shape[0] * ratio), :]
vecs = vecs / (np.linalg.norm(vecs, ord=2, axis=0, keepdims=True) + 1e-6)
qvecs = qvecs[:int(qvecs.shape[0] * ratio), :]
qvecs = qvecs / (np.linalg.norm(qvecs, ord=2, axis=0, keepdims=True) + 1e-6)
del vecs_lw, qvecs_lw
gc.collect()


# %%
# kNN search
print('>> {}: Evaluating...'.format(dataset))
vecs_T = np.zeros((vecs.shape[1], vecs.shape[0])).astype('float32')
vecs_T[:] = vecs.T[:]
import faiss  # place it in the file top will cause network load so slowly

res = faiss.StandardGpuResources()
kNN_on_gpu_id = 0
top_num = 100
k = 100

split_num = 1
for i in range(split_num):
    # scores = np.dot(vecs.T, qvecs[:, int(qvecs.shape[1]/split_num*i):int(qvecs.shape[1]/split_num*(i+1))])
    # ranks = np.argsort(-scores, axis=0)
    print('>> find nearest neighbour, k: {}, top_num: {}'.format(k, top_num))
    index = faiss.IndexFlatL2(vecs.shape[0])
    gpu_index = faiss.index_cpu_to_gpu(res, kNN_on_gpu_id, index)
    gpu_index.add(vecs_T)
    # todo: Test if this split method has bug
    query_vecs = qvecs[:, int(qvecs.shape[1] / split_num * i):int(qvecs.shape[1] / split_num * (i + 1))]
    qvecs_T = np.zeros((query_vecs.shape[1], query_vecs.shape[0])).astype('float32')
    qvecs_T[:] = query_vecs.T[:]
    _, ranks = gpu_index.search(qvecs_T, k)
    ranks = ranks.T
    if i == 0:
        ranks_top_100 = ranks[:top_num, :]
    else:
        ranks_top_100 = np.hstack((ranks_top_100, ranks[:top_num, :]))
    # del scores, ranks
    del index, query_vecs, qvecs_T
    gc.collect()
    print('\r>>>> kNN search {} nearest neighbour {}/{} done...'.format(top_num, i + 1, split_num), end='')
# del qvecs
# gc.collect()
print('')


# %%
# query expansion
QEk = 10
alpha = 10. / 2
iter = 1
QE_weight = ((np.arange(QEk, 0, -1) / QEk).reshape(1, QEk, 1) ** alpha)
ranks_split_num = 50
print('>> Query expansion: k: {}, alpha: {}, iteration: {}'.format(QEk, alpha, iter))
for i in range(ranks_split_num):
    ranks_split = ranks[:QEk, int(ranks.shape[1] / ranks_split_num * i):
                              int(ranks.shape[1] / ranks_split_num * (i + 1))]
    top_k_vecs = vecs[:, ranks_split]  # shape = (2048, QEk, query_split_size)
    qvecs_temp = (top_k_vecs * QE_weight).sum(axis=1)
    qvecs_temp = qvecs_temp / (np.linalg.norm(qvecs_temp, ord=2, axis=0, keepdims=True) + 1e-6)
    if i == 0:
        qvecs = qvecs_temp
    else:
        qvecs = np.hstack((qvecs, qvecs_temp))
    del ranks_split, top_k_vecs, qvecs_temp
    gc.collect()
    print('\r>>>> calculate new query vectors {}/{} done...'.format(i + 1, ranks_split_num), end='')
print('')
qe_iter_qvecs_path = os.path.join(get_data_root(), 'QEk{}_alpha{}_iter{}_qvecs.pkl'.format(QEk, alpha, iter))
qe_iter_qvecs_file = open(qe_iter_qvecs_path, 'wb')
pickle.dump(qvecs, qe_iter_qvecs_file)
qe_iter_qvecs_file.close()
print('>>>> QEk{}_alpha{}_iter{}_qvecs.pkl save done...'.format(QEk, alpha, iter))
# del ranks
# gc.collect()


# %%
# save to csv file
print(">> save to submission.csv file...")
submission_file = open(os.path.join(get_data_root(), 'submission.csv'), 'w')
writer = csv.writer(submission_file)
test_mark_file = open(test_mark_add_path, 'r')
csvreader = csv.reader(test_mark_file)
cnt = 0
writer.writerow(['id', 'images'])
for index, line in enumerate(csvreader):
    (flag, img_name) = line[:2]
    if flag == '1':
        select = []
        for i in range(top_num):
            select.append(images[int(ranks_top_100[i, cnt])].split('/')[-1].split('.jpg')[0])
        cnt += 1
        writer.writerow([img_name.split('/')[-1].split('.jpg')[0], ' '.join(select)])
    else:
        # random_list = random.sample(range(0, len(images)), top_num)
        random_list = np.random.choice(len(images), top_num, replace=False)
        select = []
        for i in range(top_num):
            select.append(images[random_list[i]].split('/')[-1].split('.jpg')[0])
        writer.writerow([img_name.split('/')[-1].split('.jpg')[0], ' '.join(select)])
    if cnt % 10 == 0 or cnt == len(qimages):
        print('\r>>>> {}/{} done...'.format(cnt, len(qimages)), end='')
submission_file.close()
test_mark_file.close()
print('')
print('>> {}: elapsed time: {}'.format(dataset, htime(time.time() - start)))
