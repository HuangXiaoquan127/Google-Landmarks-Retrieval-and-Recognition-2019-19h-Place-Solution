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

def cid2filename(cid, prefix):
    """
    Creates a training image path out of its CID name
    
    Arguments
    ---------
    cid      : name of the image
    prefix   : root directory where images are saved
    
    Returns
    -------
    filename : full image filename
    """
    return os.path.join(prefix, cid[-2:], cid[-4:-2], cid[-6:-4], cid)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def imresize(img, imsize):
    img.thumbnail((imsize, imsize), Image.ANTIALIAS)
    return img

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def collate_tuples(batch):
    if len(batch) == 1:
        return [batch[0][0]], [batch[0][1]]
    return [batch[i][0] for i in range(len(batch))], [batch[i][1] for i in range(len(batch))]


################################################
# hxq added
################################################
def ParseData(data_file):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [line[:2] for line in csvreader]
    return key_url_list[1:]  # Chop off header


'''
def clear_no_exist(data_table_path, mark_table_output_path, img_check_path):
    print(">> load {} file...".format(data_table_path.split('/')[-1]))
    csvfile = open(data_table_path, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [line[:2] for line in csvreader][1:]
    csvfile.close()
    images = [os.path.join(img_check_path, key_url_list[i][0]) + '.jpg' for i in range(len(key_url_list))]

    print(">> clear {} file...".format(data_table_path.split('/')[-1]))
    images_clean = []
    savefile = open(mark_table_output_path, 'w')
    save_writer = csv.writer(savefile)
    for i in range(len(images)):
        if os.path.isfile(images[i]):
            save_writer.writerow([1, key_url_list[i][0]])
            images_clean.append(images[i])
        else:
            save_writer.writerow([0, key_url_list[i][0]])

    savefile.close()
    print(">> save {} done...".format(mark_table_output_path.split('/')[-1]))

    return images_clean
'''


def clear_no_exist(data_table_path, mark_table_output_path, miss_table_output_path, img_check_path):
    print(">> load file name form {} ...".format(img_check_path.split('/')[-1]))
    files = os.listdir(img_check_path)
    files = {files[i].split('.jpg')[0]: files[i].split('.jpg')[0] for i in range(len(files))}
    csvfile = open(data_table_path, 'r')
    csvreader = csv.reader(csvfile)
    savefile = open(mark_table_output_path, 'w')
    save_writer = csv.writer(savefile)
    missfile = open(miss_table_output_path, 'w')
    miss_writer = csv.writer(missfile)
    print(">> processing {} file...".format(mark_table_output_path.split('/')[-1]))
    miss_writer.writerow(['id', 'url'])
    for i, line in enumerate(csvreader):
        if i != 0:
            if files.__contains__(line[0]):
                save_writer.writerow(['1', line[0]])
                files.pop(line[0])
            else:
                save_writer.writerow(['0', line[0]])
                miss_writer.writerow(line)
    csvfile.close()
    savefile.close()
    missfile.close()
    print(">> save {} and {} done...".format(mark_table_output_path.split('/')[-1],
                                             miss_table_output_path.split('/')[-1]))


'''
def generate_train_val_pkl(dataset, data_table_path, img_check_path, pkl_output_path, val_set_size):
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
        missfile = open(os.path.join(pkl_output_path, 'train_miss.csv'), 'w')
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

    # split to train and val set, then sorting
    print('>> split train val set and sorting...')
    shuffle = np.random.permutation(len(train_temp))

    train = []
    train_dict = {}
    for i in shuffle[:-val_set_size]:
        train.append(train_temp[i])
    train = pd.DataFrame(train, columns=['mark', 'id', 'landmark_id'])
    train['landmark_id'] = train['landmark_id'].astype('int')
    train = train.sort_values('landmark_id')
    train = train.to_numpy()
    train_id, train_landmark_id = [], []
    for i in range(len(train)):
        if train[i, 0] == '1':
            train_id.append(train[i, 1])
            train_landmark_id.append(train[i, 2])
    train_dict['id'] = train_id
    train_dict['landmark_id'] = train_landmark_id

    val = []
    val_dict = {}
    for i in shuffle[-val_set_size:]:
        val.append(train_temp[i])
    val = pd.DataFrame(val, columns=['mark', 'id', 'landmark_id'])
    val['landmark_id'] = val['landmark_id'].astype('int')
    val = val.sort_values('landmark_id')
    val = val.to_numpy()
    val_id, val_landmark_id = [], []
    for i in range(len(val)):
        if val[i, 0] == '1':
            val_id.append(val[i, 1])
            val_landmark_id.append(val[i, 2])
    val_dict['id'] = val_id
    val_dict['landmark_id'] = val_landmark_id

    # choice specific query and positive pair num per landmark
    if dataset == 'google-landmarks-dataset-v2':
        pair_num = float('inf')
    elif dataset == 'google-landmarks-dataset-resize':
        pair_num = 50
    elif dataset == 'google-landmarks-dataset':
        pair_num = float('inf')
    else:
        pair_num = float('inf')

    # generate train_set and val_set's qidxs and pidxs
    pair_cnt = 0
    print('>> generate train and val set\'s qidxs and pidxs...')
    train_qidxs, train_pidxs = [], []
    for i in range(len(train_dict['landmark_id'])):
        if train_dict['landmark_id'][i] != -1:
            if i != len(train_dict['landmark_id']) - 1 and \
                    train_dict['landmark_id'][i] == train_dict['landmark_id'][i + 1]:
                if pair_cnt < pair_num:
                    train_qidxs.append(i)
                    train_pidxs.append(i + 1)
                    pair_cnt += 1
            else:
                pair_cnt = 0
    train_dict['qidxs'] = train_qidxs
    train_dict['pidxs'] = train_pidxs

    pair_cnt = 0
    val_qidxs, val_pidxs = [], []
    for i in range(len(val_dict['landmark_id'])):
        if val_dict['landmark_id'][i] != -1:
            if i != len(val_dict['landmark_id']) - 1 and val_dict['landmark_id'][i] == val_dict['landmark_id'][i + 1]:
                if pair_cnt < pair_num:
                    val_qidxs.append(i)
                    val_pidxs.append(i + 1)
                    pair_cnt += 1
            else:
                pair_cnt = 0
    val_dict['qidxs'] = val_qidxs
    val_dict['pidxs'] = val_pidxs

    # warp all dict to one pickle file
    all_dict = {'train': train_dict, 'val': val_dict}
    pickle_file = open(os.path.join(pkl_output_path, '{}.pkl'.format(dataset)), 'wb')
    pickle.dump(all_dict, pickle_file)
    pickle_file.close()
    print(">> save {}.pkl done...".format(dataset))
'''


'''
def gen_train_val_test_pkl(dataset, data_table_path, img_check_path, val_set_size, test_set_size,
                           train_val_file_output_path, test_file_output_path):
    # setp1: check image is exist
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

    # step2: extract test set
    print(">> extract test set...")
    shuffle = np.random.permutation(len(train_temp))
    train_shuffle = []
    for i in shuffle:
        train_shuffle.append(train_temp[i])

    train_shuffle = pd.DataFrame(train_shuffle, columns=['mark', 'id', 'landmark_id'])
    train_shuffle['landmark_id'] = train_shuffle['landmark_id'].astype('int')
    # train = train.sort_values('landmark_id')
    train_shuffle = train_shuffle.to_numpy()

    select_landmark_num = round(test_set_size / (len(train_shuffle) / max(train_shuffle[:, 2])))
    shuffle_landmark_id = np.random.permutation(max(train_shuffle[:, 2]))[:select_landmark_num]
    test = []
    for i in range(len(train_shuffle)):
        if train_shuffle[i][2] in shuffle_landmark_id:
            if train_shuffle[i][0] == '1':
                test.append(train_shuffle[i][1:])
                train_shuffle[i][0] = '0'

    test = pd.DataFrame(test, columns=['id', 'landmark_id'])
    test = test.sort_values('landmark_id')
    test = test.to_numpy()

    print(">> copy test image from train set...")
    imlist_all = list(test[:, 0])
    img_path = [os.path.join(img_check_path, imlist_all[i] + '.jpg') for i in range(len(imlist_all))]
    for i in range(len(imlist_all)):
        shutil.copyfile(img_path[i], os.path.join(test_file_output_path, 'jpg', imlist_all[i] + '.jpg'))

    test_fold = []
    for i in range(select_landmark_num):
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
    q_index_ratio = 118000. / 1100000.
    for i in range(select_landmark_num):
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

    # step3: split to train and val set
    print('>> split train val set...')

    train_dict = {}
    train = train_shuffle[:-val_set_size]
    train = pd.DataFrame(train, columns=['mark', 'id', 'landmark_id'])
    train['landmark_id'] = train['landmark_id'].astype('int')
    train = train.sort_values('landmark_id')
    train = train.to_numpy()
    train_id, train_landmark_id = [], []
    for i in range(len(train)):
        if train[i, 0] == '1':
            train_id.append(train[i, 1])
            train_landmark_id.append(train[i, 2])
    train_dict['id'] = train_id
    train_dict['landmark_id'] = train_landmark_id

    val_dict = {}
    val = train_shuffle[-val_set_size:]
    val = pd.DataFrame(val, columns=['mark', 'id', 'landmark_id'])
    val['landmark_id'] = val['landmark_id'].astype('int')
    val = val.sort_values('landmark_id')
    val = val.to_numpy()
    val_id, val_landmark_id = [], []
    for i in range(len(val)):
        if val[i, 0] == '1':
            val_id.append(val[i, 1])
            val_landmark_id.append(val[i, 2])
    val_dict['id'] = val_id
    val_dict['landmark_id'] = val_landmark_id

    # choice specific query and positive pair num per landmark
    if dataset == 'google-landmarks-dataset-v2':
        pair_num = 50
    elif dataset == 'google-landmarks-dataset-resize':
        pair_num = 50
    elif dataset == 'google-landmarks-dataset':
        pair_num = 50
    else:
        pair_num = float('inf')

    # generate train_set and val_set's qidxs and pidxs
    pair_cnt = 0
    print('>> generate train and val set\'s qidxs and pidxs...')
    train_qidxs, train_pidxs = [], []
    for i in range(len(train_dict['landmark_id'])-1):
        if train_dict['landmark_id'][i] != -1:
            if train_dict['landmark_id'][i] == train_dict['landmark_id'][i + 1]:
                if pair_cnt < pair_num:
                    train_qidxs.append(i)
                    train_pidxs.append(i + 1)
                    pair_cnt += 1
            else:
                pair_cnt = 0
    train_dict['qidxs'] = train_qidxs
    train_dict['pidxs'] = train_pidxs

    pair_cnt = 0
    val_qidxs, val_pidxs = [], []
    for i in range(len(val_dict['landmark_id'])-1):
        if val_dict['landmark_id'][i] != -1:
            if val_dict['landmark_id'][i] == val_dict['landmark_id'][i + 1]:
                if pair_cnt < pair_num:
                    val_qidxs.append(i)
                    val_pidxs.append(i + 1)
                    pair_cnt += 1
            else:
                pair_cnt = 0
    val_dict['qidxs'] = val_qidxs
    val_dict['pidxs'] = val_pidxs

    # warp all dict to one pickle file
    all_dict = {'train': train_dict, 'val': val_dict}
    pickle_file = open(os.path.join(pkl_output_path, '{}.pkl'.format(dataset)), 'wb')
    pickle.dump(all_dict, pickle_file)
    pickle_file.close()
    print(">> save {}.pkl done...".format(dataset))
'''