import os
from PIL import Image
# hxq added
import os, csv
import pandas as pd
import numpy as np
import os, csv
import pickle

import torch
import shutil
from cirtorch.utils.general import get_data_root


def gen_train_val_test_pkl(dataset):
    if dataset == 'google-landmarks-dataset-v2':
        data_table_path = os.path.join(get_data_root(), 'train.csv')
        img_check_path = os.path.join(get_data_root(), 'train')
        train_val_file_output_path = os.path.join(get_data_root())
        test_file_output_path = os.path.join(get_data_root(), 'test', dataset + '-test')
        val_set_size = 10000
        test_set_size = 10000
        q_index_ratio = 100000. / 700000.
        train_qp_pair_num = 12
    elif dataset == 'google-landmarks-dataset-resize':
        data_table_path = os.path.join(get_data_root(), 'train.csv')
        img_check_path = os.path.join(get_data_root(), 'resize_train_image')
        train_val_file_output_path = os.path.join(get_data_root())
        test_file_output_path = os.path.join(get_data_root(), 'test', dataset + '-test')
        val_set_size = 10000
        test_set_size = 10000
        q_index_ratio = 118000. / 1100000.
        train_qp_pair_num = 50
    elif dataset == 'google-landmarks-dataset':
        pass

    # step0: check image is exist
    print(">> load {} file...".format(data_table_path.split('/')[-1]))
    csvfile = open(data_table_path, 'r')
    csvreader = csv.reader(csvfile)
    id_url_class_list = [line[:3] for line in csvreader][1:]
    csvfile.close()
    if dataset == 'google-landmarks-dataset-v2':
        # https://github.com/cvdfoundation/google-landmark
        # images = [os.path.join(img_check_path, '/'.join(list(id_url_class_list[i][0])[:3]), id_url_class_list[i][0])
        #           + '.jpg' for i in range(len(id_url_class_list))]
        train_temp = [['1', id_url_class_list[i][0], id_url_class_list[i][2]] for i in range(len(id_url_class_list))]
    elif dataset == 'google-landmarks-dataset-resize':
        print(">> check {} file...".format(data_table_path.split('/')[-1]))
        train_temp = []
        files = os.listdir(img_check_path)
        files = {files[i].split('.jpg')[0]: files[i].split('.jpg')[0] for i in range(len(files))}
        missfile = open(os.path.join(train_val_file_output_path, 'train_miss.csv'), 'w')
        miss_writer = csv.writer(missfile)
        miss_writer.writerow(['id', 'url'])
        miss_cnt = 0
        for line in id_url_class_list:
            if files.__contains__(line[0]):
                if line[2] != 'None':
                    train_temp.append(['1', line[0], line[2]])
                else:
                    train_temp.append(['1', line[0], '-1'])
                files.pop(line[0])
            else:
                if line[2] != 'None':
                    train_temp.append(['0', line[0], line[2]])
                else:
                    train_temp.append(['0', line[0], '-1'])
                miss_writer.writerow(line)
                miss_cnt += 1
        missfile.close()
        print('>>>> train image miss: {}, train_miss.csv save done...'.format(miss_cnt))
    elif dataset == 'google-landmarks-dataset':
        pass

    np.random.seed(0)  # added after the competition
    shuffle = np.random.permutation(len(train_temp))
    train_shuffle = []
    for i in shuffle:
        train_shuffle.append(train_temp[i])

    train_shuffle = pd.DataFrame(train_shuffle, columns=['mark', 'id', 'landmark_id'])
    train_shuffle['landmark_id'] = train_shuffle['landmark_id'].astype('int')
    # train = train.sort_values('landmark_id')
    train_shuffle = train_shuffle.to_numpy()

    # step1: extract test set
    print(">> extract test set...")
    landmark_id_shuffle = np.random.permutation(max(train_shuffle[:, 2]))
    test_landmark_num = round(test_set_size / (len(train_shuffle) / max(train_shuffle[:, 2])))
    test_sel_landmark_id = landmark_id_shuffle[:test_landmark_num]
    landmark_id_shuffle = landmark_id_shuffle[test_landmark_num:]
    test = []
    for i in range(len(train_shuffle)):
        if train_shuffle[i][2] in test_sel_landmark_id:
            if train_shuffle[i][0] == '1':
                test.append(train_shuffle[i][1:])
                train_shuffle[i][0] = '0'

    test = pd.DataFrame(test, columns=['id', 'landmark_id'])
    test = test.sort_values('landmark_id')
    test = test.to_numpy()

    print(">> copy test image from train set...")
    imlist_all = list(test[:, 0])
    if dataset == 'google-landmarks-dataset-v2':
        img_path = [os.path.join(img_check_path, '/'.join(list(imlist_all[i])[:3]), imlist_all[i] + '.jpg')
                    for i in range(len(imlist_all))]
    else:
        img_path = [os.path.join(img_check_path, imlist_all[i] + '.jpg') for i in range(len(imlist_all))]
    for i in range(len(imlist_all)):
        shutil.copyfile(img_path[i], os.path.join(test_file_output_path, 'jpg', imlist_all[i] + '.jpg'))

    test_fold = []
    for i in range(test_landmark_num):
        test_fold.append([])
    landmark_cnt = 0
    test_fold[0].append([test[0][0], 0])
    for i in range(1, len(test)):
        if test[i][1] != test[i-1][1]:
            landmark_cnt += 1
        test_fold[landmark_cnt].append([test[i][0], i])

    qimlist = []
    gnd = []
    imlist = []
    for i in range(test_landmark_num):
        q_num = max(round(q_index_ratio * len(test_fold[i])), 1)
        for j in range(q_num):
            qimlist.append(test_fold[i][j][0])
            gnd.append({'ok': []})
            for k in range(len(imlist), len(imlist) + len(test_fold[i]) - q_num):
                gnd[-1]['ok'].append(k)
        for j in range(q_num, len(test_fold[i])):
            imlist.append(test_fold[i][j][0])

    test_pkl = {'imlist': imlist, 'qimlist': qimlist, 'gnd': gnd}
    pkl_output_path = os.path.join(test_file_output_path, 'gnd_{}-test.pkl'.format(dataset))
    pickle_file = open(pkl_output_path, 'wb')
    pickle.dump(test_pkl, pickle_file)
    pickle_file.close()
    print(">> save gnd_{}-test.pkl done...".format(dataset))

    # step2: extract validation set
    print('>> extract validation set...')
    val_landmark_num = round(val_set_size / (len(train_shuffle) / max(train_shuffle[:, 2])))
    val_sel_landmark_id = landmark_id_shuffle[:val_landmark_num]
    val = []
    for i in range(len(train_shuffle)):
        if train_shuffle[i][2] in val_sel_landmark_id:
            if train_shuffle[i][0] == '1':
                val.append(train_shuffle[i][1:])
                train_shuffle[i][0] = '0'
    val = pd.DataFrame(val, columns=['id', 'landmark_id'])
    val = val.sort_values('landmark_id')
    val = val.to_numpy()
    val_dict = {}
    val_dict['id'] = list(val[:, 0])
    val_dict['landmark_id'] = list(val[:, 1])

    pair_cnt = 0
    val_qidxs, val_pidxs = [], []
    val_qp_pair_num = round(train_qp_pair_num * (val_set_size / (len(train_shuffle) - val_set_size - test_set_size)))
    for i in range(len(val_dict['landmark_id'])-1):
        if val_dict['landmark_id'][i] != -1:
            if val_dict['landmark_id'][i] == val_dict['landmark_id'][i + 1]:
                if pair_cnt < max(1, val_qp_pair_num):
                    val_qidxs.append(i)
                    val_pidxs.append(i + 1)
                    pair_cnt += 1
            else:
                pair_cnt = 0
    val_dict['qidxs'] = val_qidxs
    val_dict['pidxs'] = val_pidxs

    if dataset == 'google-landmarks-dataset-v2':
        print(">> copy val image from train set...")
        img_path = [os.path.join(img_check_path, '/'.join(list(val_dict['id'][i])[:3]), val_dict['id'][i] + '.jpg')
                    for i in range(len(val_dict['id']))]
        for i in range(len(val_dict['id'])):
            shutil.copyfile(img_path[i], os.path.join(train_val_file_output_path, 'val', val_dict['id'][i] + '.jpg'))

    # step3: extract train set
    print('>> extract train set...')

    train_dict = {}
    train = train_shuffle
    train = pd.DataFrame(train, columns=['mark', 'id', 'landmark_id'])
    train = train.sort_values('landmark_id')
    train = train.to_numpy()
    train_id, train_landmark_id = [], []
    for i in range(len(train)):
        if train[i, 0] == '1':
            train_id.append(train[i, 1])
            train_landmark_id.append(train[i, 2])
    train_dict['id'] = train_id
    train_dict['landmark_id'] = train_landmark_id

    pair_cnt = 0
    train_qidxs, train_pidxs = [], []
    for i in range(len(train_dict['landmark_id'])-1):
        if train_dict['landmark_id'][i] != -1:
            if train_dict['landmark_id'][i] == train_dict['landmark_id'][i + 1]:
                if pair_cnt < train_qp_pair_num:
                    train_qidxs.append(i)
                    train_pidxs.append(i + 1)
                    pair_cnt += 1
            else:
                pair_cnt = 0
    train_dict['qidxs'] = train_qidxs
    train_dict['pidxs'] = train_pidxs

    # warp all dict to one pickle file
    all_dict = {'train': train_dict, 'val': val_dict}
    pickle_file = open(os.path.join(train_val_file_output_path, '{}.pkl'.format(dataset)), 'wb')
    pickle.dump(all_dict, pickle_file)
    pickle_file.close()
    print(">> save {}.pkl done...".format(dataset))
