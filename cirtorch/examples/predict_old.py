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


PRETRAINED = {
    'retrievalSfM120k-vgg16-gem'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth',
    'retrievalSfM120k-resnet101-gem'    : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
}

datasets_names = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k',
                  'google-landmarks-dataset-resize', 'google-landmarks-dataset', 'google-landmarks-dataset-v2']
whitening_names = ['retrieval-SfM-30k', 'retrieval-SfM-120k']

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Testing')

# network
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--network-path', '-npath', metavar='NETWORK',
                    help="network path, destination where network is saved")
group.add_argument('--network-offtheshelf', '-noff', metavar='NETWORK',
                    help="network off-the-shelf, in the format 'ARCHITECTURE-POOLING' or 'ARCHITECTURE-POOLING-{reg-lwhiten-whiten}'," + 
                        " examples: 'resnet101-gem' | 'resnet101-gem-reg' | 'resnet101-gem-whiten' | 'resnet101-gem-lwhiten' | 'resnet101-gem-reg-whiten'")

# test options
parser.add_argument('--datasets', '-d', metavar='DATASETS', default='oxford5k,paris6k',
                    help="comma separated list of test datasets: " + 
                        " | ".join(datasets_names) + 
                        " (default: 'oxford5k,paris6k')")
parser.add_argument('--image-size', '-imsize', default=1024, type=int, metavar='N',
                    help="maximum size of longer image side used for testing (default: 1024)")
parser.add_argument('--multiscale', '-ms', metavar='MULTISCALE', default='[1]', 
                    help="use multiscale vectors for testing, " + 
                    " examples: '[1]' | '[1, 1/2**(1/2), 1/2]' | '[1, 2**(1/2), 1/2**(1/2)]' (default: '[1]')")
parser.add_argument('--whitening', '-w', metavar='WHITENING', default=None, choices=whitening_names,
                    help="dataset used to learn whitening for testing: " + 
                        " | ".join(whitening_names) + 
                        " (default: None)")

# GPU ID
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                    help="gpu id used for testing (default: '0')")

# parser.add_argument('--output-path', metavar='EXPORT_DIR',
#                     help='destination where mistake predict image should be saved')


def main():
    args = parser.parse_args()

    # check if there are unknown datasets
    for dataset in args.datasets.split(','):
        if dataset not in datasets_names:
            raise ValueError('Unsupported or unknown dataset: {}!'.format(dataset))

    # check if test dataset are downloaded
    # and download if they are not
    # download_train(get_data_root())
    # download_test(get_data_root())

    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # loading network from path
    if args.network_path is not None:

        print(">> Loading network:\n>>>> '{}'".format(args.network_path))
        if args.network_path in PRETRAINED:
            # pretrained networks (downloaded automatically)
            state = load_url(PRETRAINED[args.network_path], model_dir=os.path.join(get_data_root(), 'networks'))
        else:
            # fine-tuned network from path
            state = torch.load(args.network_path)

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

        # load network
        net = init_network(net_params)
        net.load_state_dict(state['state_dict'])
        
        # if whitening is precomputed
        if 'Lw' in state['meta']:
            net.meta['Lw'] = state['meta']['Lw']
        
        print(">>>> loaded network: ")
        print(net.meta_repr())

    # loading offtheshelf network
    elif args.network_offtheshelf is not None:
        
        # parse off-the-shelf parameters
        offtheshelf = args.network_offtheshelf.split('-')
        net_params = {}
        net_params['architecture'] = offtheshelf[0]
        net_params['pooling'] = offtheshelf[1]
        net_params['local_whitening'] = 'lwhiten' in offtheshelf[2:]
        net_params['regional'] = 'reg' in offtheshelf[2:]
        net_params['whitening'] = 'whiten' in offtheshelf[2:]
        net_params['pretrained'] = True

        # load off-the-shelf network
        print(">> Loading off-the-shelf network:\n>>>> '{}'".format(args.network_offtheshelf))
        net = init_network(net_params)
        print(">>>> loaded network: ")
        print(net.meta_repr())

    # setting up the multi-scale parameters
    ms = list(eval(args.multiscale))
    if len(ms)>1 and net.meta['pooling'] == 'gem' and not net.meta['regional'] and not net.meta['whitening']:
        msp = net.pool.p.item()
        print(">> Set-up multiscale:")
        print(">>>> ms: {}".format(ms))            
        print(">>>> msp: {}".format(msp))
    else:
        msp = 1

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

    # evaluate on test datasets
    datasets = args.datasets.split(',')
    for dataset in datasets: 
        start = time.time()
        print('>> {}: Extracting...'.format(dataset))

        print('>> Prepare data information...')
        index_file_path = os.path.join(get_data_root(), 'index.csv')
        index_mark_path = os.path.join(get_data_root(), 'index_mark.csv')
        index_miss_path = os.path.join(get_data_root(), 'index_miss.csv')
        test_file_path = os.path.join(get_data_root(), 'test.csv')
        test_mark_path = os.path.join(get_data_root(), 'test_mark.csv')
        test_mark_add_path = os.path.join(get_data_root(), 'test_mark_add.csv')
        test_miss_path = os.path.join(get_data_root(), 'test_miss.csv')
        if dataset == 'google-landmarks-dataset':
            index_img_path = os.path.join(get_data_root(), 'index')
            test_img_path = os.path.join(get_data_root(), 'google-landmarks-dataset-test')
        elif dataset == 'google-landmarks-dataset-resize':
            index_img_path = os.path.join(get_data_root(), 'resize_index_image')
            test_img_path = os.path.join(get_data_root(), 'resize_test_image')
        if not (os.path.isfile(index_mark_path) or os.path.isfile(index_miss_path)):
            clear_no_exist(index_file_path, index_mark_path, index_miss_path, index_img_path)
        if not (os.path.isfile(test_mark_path) or os.path.isfile(test_miss_path)):
            clear_no_exist(test_file_path, test_mark_path, test_miss_path, test_img_path)

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
        print('>>>> index image miss: {}, supplement: {}, still miss: {}'.format(miss, add, miss-add))

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

        # extract index vectors
        print('>> {}: index images...'.format(dataset))
        split_num = 6
        extract_num = int(len(images) / split_num)
        num_list = list(range(0, len(images), extract_num))
        num_list.append(len(images))

        # k = 0
        # print('>>>> extract part {} of {}'.format(k, split_num-1))
        # vecs = extract_vectors(net, images[num_list[k]:num_list[k+1]], args.image_size, transform, ms=ms, msp=msp)
        # vecs = vecs.numpy()
        # print('>>>> save index vecs to pkl...')
        # vecs_file_path = os.path.join(get_data_root(), 'index_vecs{}_of_{}.pkl'.format(k+1, split_num))
        # vecs_file = open(vecs_file_path, 'wb')
        # pickle.dump(vecs[:, num_list[k]:num_list[k+1]], vecs_file)
        # vecs_file.close()
        # print('>>>> index_vecs{}_of_{}.pkl save done...'.format(k+1, split_num))

        for i in range(split_num):
            # vecs_temp = np.loadtxt(open(os.path.join(get_data_root(), 'index_vecs{}_of_{}.csv'.format(i+1, split_num)), "rb"),
            #                        delimiter=",", skiprows=0)
            with open(os.path.join(get_data_root(), 'index_vecs{}_of_{}.pkl'.format(i+1, split_num)), 'rb') as f:
                vecs_temp = pickle.load(f)
            if i == 0:
                vecs = vecs_temp
            else:
                vecs = np.hstack((vecs, vecs_temp[:, :]))
            del vecs_temp
            gc.collect()
            print('\r>>>> index_vecs{}_of_{}.pkl load done...'.format(i+1, split_num), end='')
        print('')

        # extract query vectors
        print('>> {}: query images...'.format(dataset))
        split_num = 1
        extract_num = int(len(qimages) / split_num)
        num_list = list(range(0, len(qimages), extract_num))
        num_list.append(len(qimages))
        # k = 0
        # print('>>>> extract part {} of {}'.format(k, split_num - 1))
        # qvecs = extract_vectors(net, qimages[num_list[k]:num_list[k + 1]], args.image_size, transform, ms=ms, msp=msp)
        # qvecs = qvecs.numpy()
        # for i in range(split_num):
        #     qvecs_file_path = os.path.join(get_data_root(), 'test_vecs{}_of_{}.pkl'.format(i+1, split_num))
        #     qvecs_file = open(qvecs_file_path, 'wb')
        #     pickle.dump(qvecs[:, num_list[i]:num_list[i+1]], qvecs_file)
        #     qvecs_file.close()
        #     print('\r>>>> test_vecs{}_of_{}.pkl save done...'.format(i+1, split_num), end='')
        # print('')

        for i in range(split_num):
            # qvecs_temp = np.loadtxt(open(os.path.join(get_data_root(), 'test_vecs{}_of_{}.csv'.format(i+1, split_num)), "rb"),
            #                         delimiter=",", skiprows=0)
            with open(os.path.join(get_data_root(), 'test_vecs{}_of_{}.pkl'.format(i+1, split_num)), 'rb') as f:
                qvecs_temp = pickle.load(f)
            if i == 0:
                qvecs = qvecs_temp
            else:
                qvecs = np.hstack((qvecs, qvecs_temp[:, :]))
            del qvecs_temp
            gc.collect()
            print('\r>>>> test_vecs{}_of_{}.pkl load done...'.format(i+1, split_num), end='')
        print('')

        # vecs = np.zeros((2048, 1093278))
        # qvecs = np.zeros((2048, 115921))

        # save vecs to csv file
        # np.savetxt(os.path.join(get_data_root(), 'index_vecs{}_of_{}.csv'.format(k, split_num-1)), vecs, delimiter=',')
        # np.savetxt(os.path.join(get_data_root(), 'test_vecs{}_of_{}.csv'.format(k, split_num-1)), qvecs, delimiter=',')

        # compute whitening
        if args.whitening is not None:
            start = time.time()
            if 'Lw' in net.meta and args.whitening in net.meta['Lw']:
                print('>> {}: Whitening is precomputed, loading it...'.format(args.whitening))
                if len(ms) > 1:
                    Lw = net.meta['Lw'][args.whitening]['ms']
                else:
                    Lw = net.meta['Lw'][args.whitening]['ss']
            else:
                # if we evaluate networks from path we should save/load whitening
                # not to compute it every time
                if args.network_path is not None:
                    whiten_fn = args.network_path + '_{}_whiten'.format(args.whitening)
                    if len(ms) > 1:
                        whiten_fn += '_ms'
                    whiten_fn += '.pth'
                else:
                    whiten_fn = None
                if whiten_fn is not None and os.path.isfile(whiten_fn):
                    print('>> {}: Whitening is precomputed, loading it...'.format(args.whitening))
                    Lw = torch.load(whiten_fn)
                else:
                    print('>> {}: Learning whitening...'.format(args.whitening))
                    # extract whitening vectors
                    print('>> {}: Extracting...'.format(args.whitening))
                    wvecs = vecs
                    # learning whitening
                    print('>> {}: Learning...'.format(args.whitening))
                    m, P = pcawhitenlearn(wvecs)
                    Lw = {'m': m, 'P': P}
                    # saving whitening if whiten_fn exists
                    if whiten_fn is not None:
                        print('>> {}: Saving to {}...'.format(args.whitening, whiten_fn))
                        torch.save(Lw, whiten_fn)
            print('>> {}: elapsed time: {}'.format(args.whitening, htime(time.time() - start)))
        else:
            Lw = None

        print('>> apply PCAwhiten...')
        if Lw is not None:
            # whiten the vectors and shrinkage
            vecs  = whitenapply(vecs, Lw['m'], Lw['P'])
            qvecs = whitenapply(qvecs, Lw['m'], Lw['P'])

        print('>>>> save index PCAwhiten vecs to pkl...')
        split_num = 6
        extract_num = int(len(images) / split_num)
        num_list = list(range(0, len(images), extract_num))
        num_list.append(len(images))
        for i in range(split_num):
            vecs_file_path = os.path.join(get_data_root(), 'index_PCAwhiten_vecs{}_of_{}.pkl'.format(i+1, split_num))
            vecs_file = open(vecs_file_path, 'wb')
            pickle.dump(vecs[:, num_list[i]:num_list[i+1]], vecs_file)
            vecs_file.close()
            print('\r>>>> index_PCAwhiten_vecs{}_of_{}.pkl save done...'.format(i+1, split_num), end='')
        print('')

        print('>>>> save test PCAwhiten vecs to pkl...')
        split_num = 1
        extract_num = int(len(qimages) / split_num)
        num_list = list(range(0, len(qimages), extract_num))
        num_list.append(len(images))
        for i in range(split_num):
            qvecs_file_path = os.path.join(get_data_root(), 'test_PCAwhiten_vecs{}_of_{}.pkl'.format(i+1, split_num))
            qvecs_file = open(qvecs_file_path, 'wb')
            pickle.dump(qvecs[:, num_list[i]:num_list[i+1]], qvecs_file)
            qvecs_file.close()
            print('\r>>>> test_PCAwhiten_vecs{}_of_{}.pkl save done...'.format(i+1, split_num), end='')
        print('')

        print('>>>> load index PCAwhiten vecs from pkl...')
        for i in range(split_num):
            with open(os.path.join(get_data_root(), 'index_PCAwhiten_vecs{}_of_{}.pkl'.format(i+1, split_num)), 'rb') as f:
                vecs_temp = pickle.load(f)
            if i == 0:
                vecs = vecs_temp
            else:
                vecs = np.hstack((vecs, vecs_temp[:, :]))
            del vecs_temp
            gc.collect()
            print('\r>>>> index_PCAwhiten_vecs{}_of_{}.pkl load done...'.format(i+1, split_num), end='')
        print('')

        print('>>>> load test PCAwhiten vecs from pkl...')
        for i in range(split_num):
            with open(os.path.join(get_data_root(), 'test_PCAwhiten_vecs{}_of_{}.pkl'.format(i+1, split_num)), 'rb') as f:
                qvecs_temp = pickle.load(f)
            if i == 0:
                qvecs = qvecs_temp
            else:
                qvecs = np.hstack((qvecs, qvecs_temp[:, :]))
            del qvecs_temp
            gc.collect()
            print('\r>>>> test_PCAwhiten_vecs{}_of_{}.pkl load done...'.format(i+1, split_num), end='')
        print('')

        # extract principal components and dimension shrinkage
        ratio = 0.8
        vecs  = vecs[:int(vecs.shape[0]*ratio), :]
        qvecs = vecs[:int(qvecs.shape[0]*ratio), :]

        print('>> {}: Evaluating...'.format(dataset))
        split_num = 50
        top_num = 100
        vecs_T = np.zeros((vecs.shape[1], vecs.shape[0])).astype('float32')
        vecs_T[:] = vecs.T[:]
        QE_iter = 0
        QE_weight = (np.arange(top_num, 0, -1) / top_num).reshape(1, top_num, 1)
        print('>> find {} nearest neighbour...'.format(top_num))
        import faiss  # place it in the file top will cause network load so slowly

        # ranks_top_100 = np.loadtxt(open(os.path.join(get_data_root(), 'ranks_top_{}.csv'.format(top_num)), "rb"),
        #            delimiter=",", skiprows=0).astype('int')

        for iter in range(0, QE_iter+1):
            if iter != 0:
                # ranks_top_100 = np.ones((100, 115921)).astype('int')
                print('>> Query expansion iteration {}'.format(iter))
                ranks_split = 50
                for i in range(ranks_split):
                    ranks_top_100_split = ranks_top_100[:, int(ranks_top_100.shape[1] / ranks_split * i):
                                                           int(ranks_top_100.shape[1] / ranks_split * (i + 1))]
                    top_100_vecs = vecs[:, ranks_top_100_split]  # (2048, 100, query_split_size)
                    qvecs_temp = (top_100_vecs * QE_weight).sum(axis=1)
                    qvecs_temp = qvecs_temp / (np.linalg.norm(qvecs_temp, ord=2, axis=0, keepdims=True) + 1e-6)
                    if i == 0:
                        qvecs = qvecs_temp
                    else:
                        qvecs = np.hstack((qvecs, qvecs_temp))
                    del ranks_top_100_split, top_100_vecs, qvecs_temp
                    gc.collect()
                    print('\r>>>> calculate new query vectors {}/{} done...'.format(i+1, ranks_split), end='')
                print('')
                qe_iter_qvecs_path = os.path.join(get_data_root(), 'QE_iter{}_qvecs.pkl'.format(iter))
                qe_iter_qvecs_file = open(qe_iter_qvecs_path, 'wb')
                pickle.dump(qvecs, qe_iter_qvecs_file)
                qe_iter_qvecs_file.close()
                print('>>>> QE_iter{}_qvecs.pkl save done...'.format(iter))
                del ranks_top_100
                gc.collect()
            for i in range(split_num):
                # scores = np.dot(vecs.T, qvecs[:, int(qvecs.shape[1]/split_num*i):int(qvecs.shape[1]/split_num*(i+1))])
                # ranks = np.argsort(-scores, axis=0)

                # kNN search
                k = top_num
                index = faiss.IndexFlatL2(vecs.shape[0])
                index.add(vecs_T)
                query_vecs = qvecs[:, int(qvecs.shape[1] / split_num * i):int(qvecs.shape[1] / split_num * (i + 1))]
                qvecs_T = np.zeros((query_vecs.shape[1], query_vecs.shape[0])).astype('float32')
                qvecs_T[:] = query_vecs.T[:]
                _, ranks = index.search(qvecs_T, k)
                ranks = ranks.T
                if i == 0:
                    ranks_top_100 = ranks[:top_num, :]
                else:
                    ranks_top_100 = np.hstack((ranks_top_100, ranks[:top_num, :]))
                # del scores, ranks
                del index, query_vecs, qvecs_T, ranks
                gc.collect()
                print('\r>>>> kNN search {} nearest neighbour {}/{} done...'.format(top_num, i + 1, split_num), end='')
            del qvecs
            gc.collect()
            print('')
        del vecs, vecs_T
        gc.collect()

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
        print('>> {}: elapsed time: {}'.format(dataset, htime(time.time()-start)))


if __name__ == '__main__':
    main()