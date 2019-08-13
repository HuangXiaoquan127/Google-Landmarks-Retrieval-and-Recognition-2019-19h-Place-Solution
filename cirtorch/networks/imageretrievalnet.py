import os
import pdb

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torchvision

from cirtorch.layers.pooling import MAC, SPoC, GeM, RMAC, Rpool, ConvPool, BoostGeM, LearnPool, MultiPool
from cirtorch.layers.normalization import L2N, PowerLaw
from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.utils.general import get_data_root

# hxq added
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


# for some models, we have imported features (convolutions) from caffe because the image retrieval performance is higher for them
FEATURES = {
    'vgg16'         : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-vgg16-features-d369c8e.pth',
    'resnet50'      : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet50-features-ac468af.pth',
    'resnet101'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet101-features-10a101d.pth',
    # 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet152-features-1011020.pth',
    # 'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth'

}

# TODO: pre-compute for more architectures and properly test variations (pre l2norm, post l2norm)
# pre-computed local pca whitening that can be applied before the pooling layer
L_WHITENING = {
    'resnet101' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-lwhiten-9f830ef.pth', # no pre l2 norm
    # 'resnet101' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-lwhiten-da5c935.pth', # with pre l2 norm
}


# possible global pooling layers, each on of these can be made regional
# hxq modified
POOLING = {
    'mac'  : MAC,
    'spoc' : SPoC,
    'gem'  : GeM,
    'rmac' : RMAC,
    'convpool' : ConvPool,
    'boostgem' : BoostGeM,
    'learnpool': LearnPool,
    'multipool': MultiPool,
}

# TODO: pre-compute for: resnet50-gem-r, resnet50-mac-r, vgg16-mac-r, alexnet-mac-r
# pre-computed regional whitening, for most commonly used architectures and pooling methods
R_WHITENING = {
    'alexnet-gem-r'   : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-alexnet-gem-r-rwhiten-c8cf7e2.pth',
    'vgg16-gem-r'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-vgg16-gem-r-rwhiten-19b204e.pth',
    'resnet101-mac-r' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-mac-r-rwhiten-7f1ed8c.pth',
    'resnet101-gem-r' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-gem-r-rwhiten-adace84.pth',
}

# TODO: pre-compute for more architectures
# pre-computed final (global) whitening, for most commonly used architectures and pooling methods
WHITENING = {
    'alexnet-gem'       : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-alexnet-gem-whiten-454ad53.pth',
    'alexnet-gem-r'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-alexnet-gem-r-whiten-4c9126b.pth',
    'vgg16-gem'         : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-vgg16-gem-whiten-eaa6695.pth',
    'vgg16-gem-r'       : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-vgg16-gem-r-whiten-83582df.pth',
    'resnet101-mac-r'   : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-mac-r-whiten-9df41d3.pth',
    # 'resnet101-gem'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-gem-whiten-22ab0c1.pth',
    'resnet101-gem-r'   : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-gem-r-whiten-b379c0a.pth',
}

# output dimensionality for supported architectures
OUTPUT_DIM = {
    'alexnet'       :  256,
    'vgg11'         :  512,
    'vgg13'         :  512,
    'vgg16'         :  512,
    'vgg19'         :  512,
    'resnet18'      :  512,
    'resnet34'      :  512,
    'resnet50'      : 2048,
    'resnet101'     : 2048,
    'resnet152'     : 2048,
    'densenet121'   : 1024,
    'densenet161'   : 2208,
    'densenet169'   : 1664,
    'densenet201'   : 1920,
    'squeezenet1_0' :  512,
    'squeezenet1_1' :  512,
    'resnext101_32x8d': 2048,
}


class ImageRetrievalNet(nn.Module):
    
    def __init__(self, features, lwhiten, pool, whiten, meta):
        super(ImageRetrievalNet, self).__init__()
        self.features = nn.Sequential(*features)

        # hxq added
        # self.features = nn.DataParallel(self.features)

        self.lwhiten = lwhiten
        self.pool = pool
        self.whiten = whiten
        self.norm = L2N()
        self.meta = meta
    
    def forward(self, x):
        # hxq added, for visualization test
        # x.size() is equal [N, C, H, W]
        # view channel picture
        # for c in range(x.size()[1]):
        #     plt.imshow(x[0][c].cpu().numpy())
        #     plt.show()
        # view picture
        # (C, H, W) = x[0].cpu().numpy().shape
        # plt.imshow(x[0].cpu().numpy().reshape(C, -1).transpose().reshape(H, W, C))
        # plt.show()

        # x -> features
        # o.size() is equal [N, C, H, W]
        if self.meta['multi_layer_cat'] == 1:
            o = self.features(x)
            # o_1 = self.features(x)
            pass
        elif self.meta['multi_layer_cat'] == 2:
            o2 = self.features[:-1](x)
            o = self.features[-1](o2)
            # o2_1 = self.features[:-1](x)
            # o_1 = self.features[-1](o2_1)
            pass
        elif self.meta['multi_layer_cat'] == 4:
            o4 = self.features[:-1](x)
            o3 = list(self.features[-1].children())[0](o4)
            o2 = list(self.features[-1].children())[1](o3)
            o = list(self.features[-1].children())[2](o2)
            pass

        # hxq added, output features map
        # o_fm = self.norm(o[0].reshape(2048, -1).permute(1, 0)).permute(1, 0)
        # hxq added
        # o2 = self.features[:-1](x)
        # o3 = self.features[:-2](x)

        # hxq added, overlap for attention
        # o_2 = o_1.clone()
        # overlap = o_2.sum(1, keepdim=True)
        # ratio = (overlap / overlap.max())

        # power method
        # ratio = torch.pow(ratio, 1.8)

        # sigmoid method
        # ratio = torch.sigmoid(ratio * 12 - 6) * 2
        # ratio = torch.sigmoid(ratio * 12 - 6)

        # tanh method
        # ratio = torch.tanh(ratio * 4 - 2) + 1

        # topk method
        # (_, indices) = torch.topk(ratio.reshape(1, -1), min(50, ratio.numel()), sorted=False)
        # ratio = torch.zeros(1, ratio.numel())
        # for i in indices:
        #     ratio[0,i] = 1
        # ratio = ratio.reshape(overlap.size()).cuda()

        # Peak Over Threshold method
        # ratio = ratio * (ratio >= 0.65).type_as(ratio)

        # attention version 2, CAM + overlap
        # o_2 = o_1.clone()
        # o_2 = torch.nn.Conv2d(o_2.size()[1], o_2.size()[1], kernel_size=1).cuda()(o_2)
        # o_2 = torch.nn.Conv2d(o_2.size()[1], 1, kernel_size=1).cuda()(o_2)
        # score = (o_2 / o_2.max())
        # o = o_1 * score
        #
        # o2_2 = o2_1.clone()
        # o2_2 = torch.nn.Conv2d(o2_2.size()[1], o2_2.size()[1], kernel_size=1).cuda()(o2_2)
        # o2_2 = torch.nn.Conv2d(o2_2.size()[1], 1, kernel_size=1).cuda()(o2_2)
        # score = (o2_2 / o2_2.max())
        # o2 = o2_1 * score

        # o_2 = o_1.clone()
        # o_2 = torch.nn.Conv2d(o_2.size()[1], 100, kernel_size=1).cuda()(o_2)
        # o_2 = torch.nn.Conv2d(100, 1, kernel_size=1).cuda()(o_2)
        # score = (o_2 / o_2.max())
        # o = o_1 * score
        #
        # o2_2 = o2_1.clone()
        # o2_2 = torch.nn.Conv2d(o2_2.size()[1], 100, kernel_size=1).cuda()(o2_2)
        # o2_2 = torch.nn.Conv2d(100, 1, kernel_size=1).cuda()(o2_2)
        # score = (o2_2 / o2_2.max())
        # o2 = o2_1 * score

        # attention version 3, channel attention

        # o_2 = o_1.clone()
        # max, _ = o_2.view(o_2.size()[0], o_2.size()[1], -1).max(dim=2, keepdim=True)
        # max = max.unsqueeze(2)
        # eps = 1e-6
        # ratio = (o_2 / (max + eps))
        # o = o_1 * ratio
        #
        # o2_2 = o2_1.clone()
        # max, _ = o2_2.view(o2_2.size()[0], o2_2.size()[1], -1).max(dim=2, keepdim=True)
        # max = max.unsqueeze(2)
        # eps = 1e-6
        # ratio = (o2_2 / (max + eps))
        # o2 = o2_1 * ratio

        # attention version 4, TopN attention
        # top_tensor_size = int(3)
        # top_tensor_size = int((o_1[0][0].numel() / 4.) ** (1 / 2))
        # (o, indices) = torch.topk(o_1.view(o_1.size()[0], o_1.size()[1], -1),
        #                           min(top_tensor_size**2, o_1[0][0].numel()), sorted=False)
        # o = o.reshape(o_1.size()[0], o_1.size()[1], top_tensor_size, top_tensor_size)

        # TODO: properly test (with pre-l2norm and/or post-l2norm)
        # if lwhiten exist: features -> local whiten
        if self.lwhiten is not None:
            # o = self.norm(o)
            s = o.size()
            o = o.permute(0,2,3,1).contiguous().view(-1, s[1])
            o = self.lwhiten(o)
            o = o.view(s[0],s[2],s[3],self.lwhiten.out_features).permute(0,3,1,2)
            # o = self.norm(o)

        # features -> pool -> norm
        # hxq modified
        # o = self.norm(self.pool(o)).squeeze(-1).squeeze(-1)
        # o = self.norm(self.pool(o, o2)).squeeze(-1).squeeze(-1)

        if self.meta['multi_layer_cat'] == 1:
            o = self.norm(self.pool(o)).squeeze(-1).squeeze(-1)
        elif self.meta['multi_layer_cat'] == 2:
            p1 = self.pool(o)
            p2 = self.pool(o2)
            o = self.norm(torch.cat((p1, p2), 1)).squeeze(-1).squeeze(-1)
        elif self.meta['multi_layer_cat'] == 4:
            p1 = self.pool(o)
            p2 = self.pool(o2)
            p3 = self.pool(o3)
            p4 = self.pool(o4)
            o = self.norm(torch.cat((p1, p2, p3, p4), 1)).squeeze(-1).squeeze(-1)

        # if whiten exist: pooled features -> whiten -> norm
        if self.whiten is not None:
            o = self.norm(self.whiten(o))

        # permute so that it is Dx1 column vector per image (DxN if many images)
        return o.permute(1, 0)
        # hxq modified
        # return torch.cat((o.permute(1, 0), o_fm), 1)

    def __repr__(self):
        tmpstr = super(ImageRetrievalNet, self).__repr__()[:-1]
        tmpstr += self.meta_repr()
        tmpstr = tmpstr + ')'
        return tmpstr

    def meta_repr(self):
        tmpstr = '  (' + 'meta' + '): dict( \n'  # + self.meta.__repr__() + '\n'
        tmpstr += '     architecture: {}\n'.format(self.meta['architecture'])
        tmpstr += '     local_whitening: {}\n'.format(self.meta['local_whitening'])
        tmpstr += '     pooling: {}\n'.format(self.meta['pooling'])
        tmpstr += '     regional: {}\n'.format(self.meta['regional'])
        tmpstr += '     whitening: {}\n'.format(self.meta['whitening'])
        tmpstr += '     outputdim: {}\n'.format(self.meta['outputdim'])
        tmpstr += '     multi_layer_cat: {}\n'.format(self.meta['multi_layer_cat'])
        tmpstr += '     mean: {}\n'.format(self.meta['mean'])
        tmpstr += '     std: {}\n'.format(self.meta['std'])
        tmpstr = tmpstr + '  )\n'
        return tmpstr


def init_network(params):

    # parse params with default values
    architecture = params.get('architecture', 'resnet101')
    local_whitening = params.get('local_whitening', False)
    pooling = params.get('pooling', 'gem')
    regional = params.get('regional', False)
    whitening = params.get('whitening', False)
    mean = params.get('mean', [0.485, 0.456, 0.406])
    std = params.get('std', [0.229, 0.224, 0.225])
    pretrained = params.get('pretrained', False)
    multi_layer_cat = params.get('multi_layer_cat', 1)

    # get output dimensionality size
    dim = OUTPUT_DIM[architecture]
    # # hxq modified, remove resnet50 con5_x for smaller receptive field
    # if architecture == 'resnet50':
    #     dim = 1024
    # else:
    #     dim = OUTPUT_DIM[architecture]

    # loading network from torchvision
    if pretrained:
        if architecture not in FEATURES:
            # initialize with network pretrained on imagenet in pytorch
            net_in = getattr(torchvision.models, architecture)(pretrained=True)
        else:
            # initialize with random weights, later on we will fill features with custom pretrained network
            net_in = getattr(torchvision.models, architecture)(pretrained=False)
    else:
        # initialize with random weights
        net_in = getattr(torchvision.models, architecture)(pretrained=False)

    # initialize features
    # take only convolutions for features,
    # always ends with ReLU to make last activations non-negative
    if architecture.startswith('alexnet'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('vgg'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('resnet'):
        features = list(net_in.children())[:-2]
        # # hxq modified, remove resnet50 con5_x for smaller receptive field
        # if architecture == 'resnet50':
        #     features = list(net_in.children())[:-3]
        # else:
        #     features = list(net_in.children())[:-2]
    elif architecture.startswith('densenet'):
        features = list(net_in.features.children())
        features.append(nn.ReLU(inplace=True))
    elif architecture.startswith('squeezenet'):
        features = list(net_in.features.children())
    elif architecture.startswith('resnext101_32x8d'):
        features = list(net_in.children())[:-2]
    else:
        raise ValueError('Unsupported or unknown architecture: {}!'.format(architecture))

    # initialize local whitening
    if local_whitening:
        lwhiten = nn.Linear(dim, dim, bias=True)
        # TODO: lwhiten with possible dimensionality reduce

        if pretrained:
            lw = architecture
            if lw in L_WHITENING:
                print(">> {}: for '{}' custom computed local whitening '{}' is used"
                    .format(os.path.basename(__file__), lw, os.path.basename(L_WHITENING[lw])))
                whiten_dir = os.path.join(get_data_root(), 'whiten')
                lwhiten.load_state_dict(model_zoo.load_url(L_WHITENING[lw], model_dir=whiten_dir))
            else:
                print(">> {}: for '{}' there is no local whitening computed, random weights are used"
                    .format(os.path.basename(__file__), lw))

    else:
        lwhiten = None
    
    # initialize pooling
    pool = POOLING[pooling]()
    # pool = POOLING[pooling](p=1.5)
    # print('>> GeM pool p: 1.5')
    
    # initialize regional pooling
    if regional:
        rpool = pool
        rwhiten = nn.Linear(dim, dim, bias=True)
        # TODO: rwhiten with possible dimensionality reduce

        if pretrained:
            rw = '{}-{}-r'.format(architecture, pooling)
            if rw in R_WHITENING:
                print(">> {}: for '{}' custom computed regional whitening '{}' is used"
                    .format(os.path.basename(__file__), rw, os.path.basename(R_WHITENING[rw])))
                whiten_dir = os.path.join(get_data_root(), 'whiten')
                rwhiten.load_state_dict(model_zoo.load_url(R_WHITENING[rw], model_dir=whiten_dir))
            else:
                print(">> {}: for '{}' there is no regional whitening computed, random weights are used"
                    .format(os.path.basename(__file__), rw))

        pool = Rpool(rpool, rwhiten)

    # initialize whitening
    output_dim = dim
    if whitening:
        if multi_layer_cat != 1:
            whiten = nn.Linear(dim+1024, output_dim, bias=True)
            # whiten_fn = '/home/iap205/Datasets/retrieval-SfM/whiten/R101_M2_120k_IS1024_MS1_WL.pth'
            # # whiten_fn = '/home/iap205/Datasets/retrieval-SfM/whiten/R101_CA_M2_120k_IS1024_MS1_WL.pth'
            # # whiten_fn = '/home/iap205/Datasets/retrieval-SfM/whiten/RX101_M2_120k_IS1024_MS1_WL.pth'
            # # whiten_fn = '/home/iap205/Datasets/retrieval-SfM/whiten/R101_M4_120k_IS1024_MS1_WL.pth'
            # # whiten_fn = '/home/iap205/Datasets/retrieval-SfM/whiten/RX101_M4_120k_IS1024_MS1_WL.pth'
            # print('>> load computed whitening \'{}\''.format(whiten_fn.split('/')[-1]))
            # Lw = torch.load(whiten_fn)
            # P = Lw['P'][:dim, :]
            # m = Lw['m']
            # P = torch.from_numpy(P).float()
            # m = torch.from_numpy(m).float()
            # whiten.weight.data = P
            # whiten.bias.data = -torch.mm(P, m).squeeze()
        else:
            # whiten = nn.Linear(dim, dim, bias=True)

            output_dim = 512
            whiten = nn.Linear(dim, output_dim, bias=True)
            nn.init.xavier_normal_(whiten.weight)
            nn.init.constant_(whiten.bias, 0)

            # hxq added, for whitening test
            # print('>> load the parameters of supervised whitening for the FC layer initialization')
            # whiten_fn = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/' \
            #             'R101_O_GL_FC/google-landmarks-dataset-resize_resnet101_gem_whiten_contrastive_m0.85_' \
            #             'adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362/' \
            #             'model_epoch114.pth.tar_google-landmarks-dataset_whiten_ms.pth'
            # whiten_fn = '/home/iap205/Datasets/google-landmarks-dataset-resize/whiten/imagenet-caffe-resnet101-' \
            #             'features-10a101d.pth_google-landmarks-dataset-test_256_whiten_MS1.pth'
            # whiten_fn = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2/whiten/imagenet-caffe-' \
            #             'resnet101-features-10a101d.pth_google-landmarks-dataset-v2-test_256_whiten_MS1.pth'
            # whiten_fn = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2/whiten/imagenet-caffe-' \
            #            'resnet101-features-10a101d.pth_google-landmarks-dataset-v2-test_1024_whiten_MS1.pth'
            # whiten_fn = '/media/iap205/Data/Export/cnnimageretrieval-pytorch/trained_network/retrieval-SfM-120k' \
            #             '_resnet101_gem_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5' \
            #             '_imsize362/model_best.pth.tar_retrieval-SfM-120k_whiten_ms.pth'
            # whiten_fn = '/home/iap205/Datasets/retrieval-SfM/whiten/R101_120k_IS1024_MS1.pth'
            # whiten_fn = '/home/iap205/Datasets/retrieval-SfM/whiten/R101_120k_IS362_MS1.pth'
            # whiten_fn = '/home/iap205/Datasets/retrieval-SfM/whiten/R101_120k_IS1024_MS1_WL.pth'
            # # whiten_fn = '/home/iap205/Datasets/retrieval-SfM/whiten/R101_120k_IS1024_MS1_WL_WL.pth'
            # whiten_fn = '/home/iap205/Datasets/retrieval-SfM/whiten/R101_CA_120k_IS1024_MS1_WL.pth'
            # whiten_fn = '/home/iap205/Datasets/retrieval-SfM/whiten/R101_top9_120k_IS1024_MS1_WL.pth'
            # whiten_fn = '/home/iap205/Datasets/retrieval-SfM/whiten/R101_top0.25_120k_IS1024_MS1_WL.pth'
            # whiten_fn = '/home/iap205/Datasets/retrieval-SfM/whiten/R101_LA_120k_IS1024_MS1_WL.pth'
            # whiten_fn = '/home/iap205/Datasets/retrieval-SfM/whiten/R101_top0.25_p1.5_120k_IS1024_MS1_WL.pth'
            # print('>> load computed whitening \'{}\''.format(whiten_fn.split('/')[-1]))
            # Lw = torch.load(whiten_fn)
            # P = Lw['P']
            # m = Lw['m']
            # P = torch.from_numpy(P).float()
            # m = torch.from_numpy(m).float()
            # whiten.weight.data = P
            # whiten.bias.data = -torch.mm(P, m).squeeze()

            # multi PCA test
            # whiten = nn.Sequential(nn.Linear(dim, dim, bias=True),
            #                        nn.Linear(dim, dim, bias=True))
            # whiten_fn = ['/home/iap205/Datasets/retrieval-SfM/whiten/R101_120k_362_IS1024_MS1_WL.pth',
            #              '/home/iap205/Datasets/retrieval-SfM/whiten/R101_120k_362_IS1024_MS1_WL_WL.pth']
            # for i in range(len(whiten)):
            #     print('>> load computed whitening \'{}\''.format(whiten_fn[i].split('/')[-1]))
            #     Lw = torch.load(whiten_fn[i])
            #     P = Lw['P']
            #     m = Lw['m']
            #     P = torch.from_numpy(P).float()
            #     m = torch.from_numpy(m).float()
            #     whiten[i].weight.data = P
            #     whiten[i].bias.data = -torch.mm(P, m).squeeze()
        # TODO: whiten with possible dimensionality reduce

        if pretrained:
            w = architecture
            if local_whitening:
                w += '-lw'
            w += '-' + pooling
            if regional:
                w += '-r'
            if w in WHITENING:
                print(">> {}: for '{}' custom computed whitening '{}' is used"
                    .format(os.path.basename(__file__), w, os.path.basename(WHITENING[w])))
                whiten_dir = os.path.join(get_data_root(), 'whiten')
                whiten.load_state_dict(model_zoo.load_url(WHITENING[w], model_dir=whiten_dir))
            else:
                print(">> {}: for '{}' there is no whitening computed, random weights are used"
                    .format(os.path.basename(__file__), w))
    else:
        whiten = None

    # create meta information to be stored in the network
    meta = {
        'architecture' : architecture, 
        'local_whitening' : local_whitening, 
        'pooling' : pooling, 
        'regional' : regional, 
        'whitening' : whitening, 
        'mean' : mean, 
        'std' : std,
        'outputdim' : output_dim,
        'multi_layer_cat': multi_layer_cat
    }

    # create a generic image retrieval network
    net = ImageRetrievalNet(features, lwhiten, pool, whiten, meta)

    # initialize features with custom pretrained network if needed
    if pretrained and architecture in FEATURES:
        print(">> {}: for '{}' custom pretrained features '{}' are used"
            .format(os.path.basename(__file__), architecture, os.path.basename(FEATURES[architecture])))
        model_dir = os.path.join(get_data_root(), 'networks')

        # hxq modified, remove resnet50 con5_x for smaller receptive field
        if architecture == 'resnet50-3':
            state_dict = model_zoo.load_url(FEATURES[architecture], model_dir=model_dir)
            for key in list(state_dict.keys()):
                if list(key)[0] == '7':
                    state_dict.pop(key)
            net.features.load_state_dict(state_dict)
        else:
            # hxq modified
            # state_dict = model_zoo.load_url(FEATURES[architecture], model_dir=model_dir)
            # state_dict_mGPU = {}
            # for key, val in state_dict.items():
            #     state_dict_mGPU['module.'+key] = val
            # net.features.load_state_dict(state_dict_mGPU)

            net.features.load_state_dict(model_zoo.load_url(FEATURES[architecture], model_dir=model_dir))

    return net


def extract_vectors(net, images, image_size, transform, bbxs=None, ms=[1], msp=1, print_freq=10):
    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # creating dataset loader
    loader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=images, imsize=image_size, bbxs=bbxs, transform=transform),
        batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )

    # extracting vectors
    with torch.no_grad():
        vecs = torch.zeros(net.meta['outputdim'], len(images))
        for i, input in enumerate(loader):
            input = input.cuda()  #input.shape = (1, 3, H, W)

            if len(ms) == 1:
                vecs[:, i] = extract_ss(net, input)
            else:
                # hxq modified
                # if max(input.shape[-1], input.shape[-2]) >= image_size:
                #     vecs[:, i] = extract_ms(net, input, ms, msp)
                # else:
                #     vecs[:, i] = extract_ss(net, input)

                vecs[:, i] = extract_ms(net, input, ms, msp)

            if (i+1) % print_freq == 0 or (i+1) == len(images):
                print('\r>>>> {}/{} done...'.format((i+1), len(images)), end='')
        print('')

        # hxq modified, test add features map for retrieval
        # vecs = []
        # for i, input in enumerate(loader):
        #     input = input.cuda()
        #
        #     if len(ms) == 1:
        #         vecs.append(extract_ss(net, input))
        #     else:
        #         vecs.append(extract_ms(net, input, ms, msp))
        #
        #     if (i+1) % print_freq == 0 or (i+1) == len(images):
        #         print('\r>>>> {}/{} done...'.format((i+1), len(images)), end='')
        # print('')

    return vecs


def extract_ss(net, input):
    return net(input).cpu().data.squeeze()


def extract_ms(net, input, ms, msp):

    v = torch.zeros(net.meta['outputdim'])

    for s in ms:
        if s == 1:
            input_t = input.clone()
        else:
            input_t = nn.functional.interpolate(input, scale_factor=s, mode='bilinear', align_corners=False)
        v += net(input_t).pow(msp).cpu().data.squeeze()

    v /= len(ms)
    v = v.pow(1. / msp)
    v /= v.norm()

    return v

    # hxq modified, test add features map with multi scale for retrieval

    # v = torch.zeros(net.meta['outputdim'])
    #
    # for i, s in enumerate(ms):
    #     if s == 1:
    #         input_t = input.clone()
    #     else:
    #         input_t = nn.functional.interpolate(input, scale_factor=s, mode='bilinear', align_corners=False)
    #         img = transforms.ToPILImage()(input_t.cpu().squeeze()).convert('RGB')
    #         img = img.resize((input.size()[-2], input.size()[-1]), Image.ANTIALIAS)
    #         input_t = torch.unsqueeze(transforms.ToTensor()(img), 0).cuda()
    #
    #     v_temp = net(input_t).pow(msp).cpu().data.squeeze()
    #
    #     if i != 0:
    #         v += v_temp
    #     else:
    #         v = v_temp
    #
    # v /= len(ms)
    # v = v.pow(1./msp)
    # v /= v.norm(dim=0)
    #
    # return v


def extract_regional_vectors(net, images, image_size, transform, bbxs=None, ms=[1], msp=1, print_freq=10):
    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # creating dataset loader
    loader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=images, imsize=image_size, bbxs=bbxs, transform=transform),
        batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )

    # extracting vectors
    with torch.no_grad():
        vecs = []
        for i, input in enumerate(loader):
            input = input.cuda()

            if len(ms) == 1:
                vecs.append(extract_ssr(net, input))
            else:
                # TODO: not implemented yet
                # vecs.append(extract_msr(net, input, ms, msp))
                raise NotImplementedError

            if (i+1) % print_freq == 0 or (i+1) == len(images):
                print('\r>>>> {}/{} done...'.format((i+1), len(images)), end='')
        print('')

    return vecs


def extract_ssr(net, input):
    return net.pool(net.features(input), aggregate=False).squeeze(0).squeeze(-1).squeeze(-1).permute(1,0).cpu().data


def extract_local_vectors(net, images, image_size, transform, bbxs=None, ms=[1], msp=1, print_freq=10):
    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # creating dataset loader
    loader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=images, imsize=image_size, bbxs=bbxs, transform=transform),
        batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )

    # extracting vectors
    with torch.no_grad():
        vecs = []
        for i, input in enumerate(loader):
            input = input.cuda()

            if len(ms) == 1:
                vecs.append(extract_ssl(net, input))
            else:
                # TODO: not implemented yet
                # vecs.append(extract_msl(net, input, ms, msp))
                raise NotImplementedError

            if (i+1) % print_freq == 0 or (i+1) == len(images):
                print('\r>>>> {}/{} done...'.format((i+1), len(images)), end='')
        print('')

    return vecs

def extract_ssl(net, input):
    return net.norm(net.features(input)).squeeze(0).view(net.meta['outputdim'], -1).cpu().data