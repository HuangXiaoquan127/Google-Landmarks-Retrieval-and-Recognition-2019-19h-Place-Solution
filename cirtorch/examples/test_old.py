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
from cirtorch.utils.whiten import whitenlearn, whitenapply
from cirtorch.utils.evaluate import compute_map_and_print
from cirtorch.utils.general import get_data_root, htime

# hxq added
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy import ndimage
import math
from cirtorch.datasets.landmarks_downloader import ParseData
import csv


PRETRAINED = {
    'retrievalSfM120k-vgg16-gem'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth',
    'retrievalSfM120k-resnet101-gem'    : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
}

datasets_names = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k', 'google-landmarks-dataset-resize-test',
                  'google-landmarks-dataset-v2-test']
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

parser.add_argument('--output-path', metavar='EXPORT_DIR',
                    help='destination where mistake predict image should be saved')


# hxq added
def gen_query_bbx_img(qimages, bbxs, dataset, width=8):

    bbx_qimages =[None]*len(qimages)
    for i in range(len(qimages)):
        im = Image.open(qimages[i])
        draw = ImageDraw.Draw(im)
        (x0, y0, x1, y1) = bbxs[i]
        for j in range(width):
            draw.rectangle([x0-j, y0-j, x1+j, y1+j], outline='yellow')
        bbx_qimages[i] = '_bbx.jpg'.join(qimages[i].split('.jpg'))
        im.save(bbx_qimages[i])
    print(">> {} qurery_bbx generation done...".format(dataset))

    return bbx_qimages


# hxq added, visualization of mismatched image
def mismatched_img_show_save(info, qimages, images, args, bbxs=None):
    # if network test in crop image bounding box,
    # then generate query bounding box image for mismatched image tuple show
    qimages_copy = qimages[:]
    if bbxs != None:
        qimages_copy = gen_query_bbx_img(qimages, bbxs, info['dataset'])

    if not os.path.isdir(args.output_path):
        os.mkdir(args.output_path)

    network_name = '@'.join(args.network_path.split('/')[-2:])
    output_path = os.path.join(args.output_path, network_name)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    if bbxs != None:
        output_path = os.path.join(output_path, 'bbxs')
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
    else:
        output_path = os.path.join(output_path, 'no_bbxs')
        if not os.path.isdir(output_path):
            os.mkdir(output_path)

    output_path = os.path.join(output_path,
                               'multiscale-' + str(list(eval(args.multiscale))) + '_' +
                               'whitening-' + str(args.whitening))
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    output_path = os.path.join(output_path, info['dataset'])
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    filename = os.path.join(output_path, 'mismatched_img_path.txt')
    log_file = open(filename, 'w')

    for i in range(len(qimages)):
        print('\r>>>> {}/{} done...'.format((i+1), len(qimages)), end='')
        false_pos_image = list(images[j] for j in info['fp'][i]['img_idxs'])
        false_neg_image = list(images[j] for j in info['fn'][i]['img_idxs'])

        col_num = 6
        row_num = 4
        plt.rcParams['figure.dpi'] = 150
        # for savefig:
        plt.subplots_adjust(left=0.02, bottom=0.01, right=0.98, top=0.95)
        # plt.tight_layout()
        # for show in sciview:
        # plt.subplots_adjust(left=0, bottom=0, right=2.5, top=1)

        # show query image
        plt.subplot(row_num, col_num, 1)
        plt.imshow(np.array(plt.imread(qimages_copy[i])))
        plt.axis('off')
        plt.title("Query{}".format(i))

        # show false positive image
        sample = false_pos_image[:(int(col_num * row_num / 2) - 1)]
        ranks = info['fp'][i]['ranks'][:(int(col_num * row_num / 2) - 1)]
        for idx, img in enumerate(sample):
            plt.subplot(row_num, col_num, idx + 2)
            plt.imshow(np.array(plt.imread(img)))
            plt.axis('off')
            plt.title('FP rank:' + str(ranks[idx]))

        # show false negative image
        # sample = np.random.choice(len(false_neg_image), min(int(col_num * row_num / 2) - 1, len(false_neg_image)),
        #                           replace=False)
        sample = false_neg_image[:-int(col_num * row_num / 2)-1:-1]
        ranks = info['fn'][i]['ranks'][:-int(col_num * row_num / 2)-1:-1]
        for idx, img in enumerate(sample):
            plt.subplot(row_num, col_num, idx + (col_num * row_num / 2) + 1)
            plt.imshow(np.array(plt.imread(img)))
            plt.axis('off')
            plt.title('FN rank:' + str(ranks[idx]))

        plt.savefig(os.path.join(output_path, 'mismatched_img_tuple' + str(i)), dpi='figure')
        # plt.show()
        # close all subplot, prevent picture residue
        plt.close('all')

        # output mismatched image filename to .txt
        log_file.write("\r\nquery image {} {}\r\n\r\n".format(i, qimages_copy[i]))
        # record false positive image path
        log_file.write('{:<7s}'.format('rank:') + 'false positive image path:' + "\r\n")
        for j in range(len(info['fp'][i]['img_idxs'])):
            log_file.write('{:<7s}'.format(str(info['fp'][i]['ranks'][j]))
                           + str(false_pos_image[j]) + "\r\n")

        log_file.write("\r\n")
        log_file.write("boundary_num = " + str(info['fp'][i]['boundary_num']) + "\r\n")
        log_file.write("\r\n")

        # record false negative image path
        log_file.write('{:<7s}'.format('rank:') + 'false negative image path:' + "\r\n")
        for j in range(len(info['fn'][i]['img_idxs'])):
            log_file.write('{:<7s}'.format(str(info['fn'][i]['ranks'][j]))
                           + str(false_neg_image[j]) + "\r\n")

    log_file.close()
    print('')


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
    print(">> image size: {}".format(args.image_size))
    ms = list(eval(args.multiscale))
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

    # compute whitening
    if args.whitening is not None:
        start = time.time()

        if 'Lw' in net.meta and args.whitening in net.meta['Lw']:
            
            print('>> {}: Whitening is precomputed, loading it...'.format(args.whitening))
            
            if len(ms)>1:
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
                
                # loading db
                db_root = os.path.join(get_data_root(), 'train', args.whitening)
                ims_root = os.path.join(db_root, 'ims')
                db_fn = os.path.join(db_root, '{}-whiten.pkl'.format(args.whitening))
                with open(db_fn, 'rb') as f:
                    db = pickle.load(f)
                images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]

                # extract whitening vectors
                print('>> {}: Extracting...'.format(args.whitening))
                wvecs = extract_vectors(net, images, args.image_size, transform, ms=ms, msp=msp)
                
                # learning whitening 
                print('>> {}: Learning...'.format(args.whitening))
                wvecs = wvecs.numpy()
                m, P = whitenlearn(wvecs, db['qidxs'], db['pidxs'])
                Lw = {'m': m, 'P': P}

                # saving whitening if whiten_fn exists
                if whiten_fn is not None:
                    print('>> {}: Saving to {}...'.format(args.whitening, whiten_fn))
                    torch.save(Lw, whiten_fn)

        print('>> {}: elapsed time: {}'.format(args.whitening, htime(time.time()-start)))

    else:
        Lw = None

    # evaluate on test datasets
    datasets = args.datasets.split(',')
    for dataset in datasets: 
        start = time.time()

        print('>> {}: Extracting...'.format(dataset))

        # prepare config structure for the test dataset
        cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))
        images = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
        qimages = [cfg['qim_fname'](cfg,i) for i in range(cfg['nq'])]
        # bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]

        print('>> not use bbxs...')
        bbxs = None

        # key_url_list = ParseData(os.path.join(get_data_root(), 'index.csv'))
        # index_image_path = os.path.join(get_data_root(), 'resize_index_image')
        # images = [os.path.join(index_image_path, key_url_list[i][0]) for i in range(len(key_url_list))]
        # key_url_list = ParseData(os.path.join(get_data_root(), 'test.csv'))
        # test_image_path = os.path.join(get_data_root(), 'resize_test_image')
        # qimages = [os.path.join(test_image_path, key_url_list[i][0]) for i in range(len(key_url_list))]
        # # bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]

        # csvfile = open(os.path.join(get_data_root(), 'index_clear.csv'), 'r')
        # csvreader = csv.reader(csvfile)
        # images = [line[:1][0] for line in csvreader]
        #
        # csvfile = open(os.path.join(get_data_root(), 'test_clear.csv'), 'r')
        # csvreader = csv.reader(csvfile)
        # qimages = [line[:1][0] for line in csvreader]

        # bbxs = None
        
        # extract database and query vectors
        print('>> {}: database images...'.format(dataset))
        vecs = extract_vectors(net, images, args.image_size, transform, ms=ms, msp=msp)
        # vecs = torch.randn(2048, 5063)
        # vecs = torch.randn(2048, 4993)

        # hxq modified
        # bbxs = None
        # print('>> set no bbxs...')
        print('>> {}: query images...'.format(dataset))
        qvecs = extract_vectors(net, qimages, args.image_size, transform, bbxs=bbxs, ms=ms, msp=msp)
        
        print('>> {}: Evaluating...'.format(dataset))

        # convert to numpy
        vecs = vecs.numpy()
        qvecs = qvecs.numpy()

        # search, rank, and print
        scores = np.dot(vecs.T, qvecs)

        # hxq modified, test add features map for retrieval
        # vecs = [vecs[i].numpy() for i in range(len(vecs))]
        # qvecs_temp = np.zeros((qvecs[0].shape[0], len(qvecs)))
        # for i in range(len(qvecs)):
        #     qvecs_temp[:, i] = qvecs[i][:, 0].numpy()
        # qvecs = qvecs_temp
        #
        # scores = np.zeros((len(vecs), qvecs.shape[-1]))
        # for i in range(len(vecs)):
        #     scores[i, :] = np.amax(np.dot(vecs[i].T, qvecs), 0)

        ranks = np.argsort(-scores, axis=0)
        mismatched_info = compute_map_and_print(dataset, ranks, cfg['gnd'], kappas=[1, 5, 10, 100])

        # hxq added
        show_false_img = False
        if show_false_img == True:
            print('>> Save mismatched image tuple...')
            for info in mismatched_info:
                mismatched_img_show_save(info, qimages, images, args, bbxs=bbxs)
    
        if Lw is not None:
            # whiten the vectors
            vecs_lw  = whitenapply(vecs, Lw['m'], Lw['P'])
            qvecs_lw = whitenapply(qvecs, Lw['m'], Lw['P'])

            # search, rank, and print
            scores = np.dot(vecs_lw.T, qvecs_lw)
            ranks = np.argsort(-scores, axis=0)
            mismatched_info = compute_map_and_print(dataset + ' + whiten', ranks, cfg['gnd'])

            # hxq added
            # show_false_img = False
            if show_false_img == True:
                print('>> Save mismatched image tuple...')
                for info in mismatched_info:
                    mismatched_img_show_save(info, qimages, images, args, bbxs=bbxs)
        
        print('>> {}: elapsed time: {}'.format(dataset, htime(time.time()-start)))


if __name__ == '__main__':
    main()