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
from cirtorch.datasets.datahelpers import clear_no_exist
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import shutil
# import faiss


#%%
train_file_path = os.path.join(get_data_root(), 'train.csv')
train_img_path = os.path.join(get_data_root(), 'train')
# test_file_path = os.path.join(get_data_root(), 'test-v2.csv')
# test_mark_add_path = os.path.join(get_data_root(), 'test-v2_mark_add.csv')
# test_img_path = os.path.join(get_data_root(), 'test-v2')
output_path = '/media/iap205/Data4T/Export_temp/landmarks_view'


#%%
print('>> load train image path...')
csvfile = open(train_file_path, 'r')
csvreader = csv.reader(csvfile)
images = []
images_dic = {}
id_landmarks = []
for i, line in enumerate(csvreader):
    if i != 0:
        images_dic[line[0]] = int(line[2])
        images.append(os.path.join(train_img_path, '/'.join(list(line[0])[:3]), line[0]+'.jpg'))
        id_landmarks.append([line[0], int(line[2])])
csvfile.close()
print('>>>> train image total {}'.format(i))

train = pd.DataFrame(id_landmarks, columns=['id', 'landmark_id'])
train = train.sort_values('landmark_id')
train = train.to_numpy()

images_fold_dic = {}
images_fold_dic[train[0][1]] = [train[0][0]]
for i in range(1, train.shape[0]):
    if train[i][1] == train[i-1][1]:
        images_fold_dic[train[i][1]].append(train[i][0])
    else:
        images_fold_dic[train[i][1]] = [train[i][0]]

#%%
col_num = 6
row_num = 4
plt.rcParams['figure.dpi'] = 150
# for savefig:
plt.subplots_adjust(left=0.02, bottom=0.01, right=0.98, top=0.95)
show_landmarks = [0]
for i in range(len(show_landmarks)):
    landmarks_image = []
    for img in images_fold_dic[show_landmarks[i]]:
        landmarks_image.append(os.path.join(train_img_path, '/'.join(list(img)[:3]), img+'.jpg'))
    # show image
    for j in range(min(col_num * row_num, len(landmarks_image))):
        plt.subplot(row_num, col_num, j+1)
        plt.imshow(np.array(plt.imread(landmarks_image[j])))
        plt.axis('off')
        # plt.title("Query{}".format(i))
    plt.savefig(os.path.join(output_path, 'landmark id:{}'.format(show_landmarks[i])), dpi='figure')
    # close all subplot, prevent picture residue
    plt.close('all')
    # plt.show()

#%%
show_landmarks = [93961]
for i in range(len(show_landmarks)):
    if not os.path.isdir(os.path.join(output_path, str(show_landmarks[i]))):
        os.mkdir(os.path.join(output_path, str(show_landmarks[i])))
    for img in images_fold_dic[show_landmarks[i]]:
        shutil.copyfile(os.path.join(train_img_path, '/'.join(list(img)[:3]), img+'.jpg'),
                        os.path.join(output_path, str(show_landmarks[i]), img+'.jpg'))
