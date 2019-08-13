import pandas as pd
import numpy as np
import os, csv
import pickle

from cirtorch.datasets.datahelpers import ParseData

# def ParseData(data_file):
#     csvfile = open(data_file, 'r')
#     csvreader = csv.reader(csvfile)
#     key_url_list = [line[:3] for line in csvreader]
#     return key_url_list[1:]  # Chop off header


def clear_no_exist(file_path, output_path, check_path):
    print(">> load {} file...".format(file_path.split('/')[-1]))
    key_url_list = ParseData(file_path)
    images = [os.path.join(check_path, key_url_list[i][0]) + '.jpg' for i in range(len(key_url_list))]

    print(">> clear {} file...".format(file_path.split('/')[-1]))
    savefile = open(output_path, 'w')
    save_writer = csv.writer(savefile)
    save_writer.writerow(['mark', 'id', 'landmark_id'])
    for i in range(len(images)):
        if os.path.isfile(images[i]):
            if key_url_list[i][2] != 'None':
                save_writer.writerow([1, key_url_list[i][0], key_url_list[i][2]])
            else:
                save_writer.writerow([1, key_url_list[i][0], '-1'])
        else:
            if key_url_list[i][2] != 'None':
                save_writer.writerow([0, key_url_list[i][0], key_url_list[i][2]])
            else:
                save_writer.writerow([0, key_url_list[i][0], '-1'])

    savefile.close()
    print(">> save {} done...".format(output_path.split('/')[-1]))


if __name__ == "__main__":
    file_path = '/home/iap205/Datasets/Google-Landmarks_Dataset_resize/train.csv'
    temp_output_path = '/home/iap205/Datasets/Google-Landmarks_Dataset_resize/train_clear.csv'
    check_path = '/home/iap205/Datasets/Google-Landmarks_Dataset_resize/resize_train_image'
    pkl_output_path = '/home/iap205/Datasets/Google-Landmarks_Dataset_resize/google-landmarks-dataset.pkl'

    val_set_size = 10000

    # clear_no_exist(file_path, temp_output_path, check_path)

    # split to train set and val set, and sorting
    print('>> Split to train set and val set, and sorting...')
    csvfile = open(temp_output_path, 'r')
    csvreader = csv.reader(csvfile)
    train_clear = [line[:3] for line in csvreader][1:]
    shuffle = np.random.permutation(len(train_clear))

    train = []
    train_dict = {}
    for i in shuffle[:-val_set_size]:
        train.append(train_clear[i])
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

    # train.to_csv(split_train_output_path, index=False)
    # print(">> save {} done...".format(split_train_output_path.split('/')[-1]))

    val = []
    val_dict = {}
    for i in shuffle[-val_set_size:]:
        val.append(train_clear[i])
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

    # val.to_csv(split_val_output_path, index=False)
    # print(">> save {} done...".format(split_val_output_path.split('/')[-1]))

    # generate train_set and val_set's qidxs and pidxs
    print('>> Generate train_set and val_set\'s qidxs and pidxs...')
    train_qidxs, train_pidxs = [], []
    for i in range(len(train_dict['landmark_id'])):
        if train_dict['landmark_id'][i] != -1:
            if i != len(train_dict['landmark_id'])-1 and train_dict['landmark_id'][i] == train_dict['landmark_id'][i+1]:
                train_qidxs.append(i)
                train_pidxs.append(i+1)
    train_dict['qidxs'] = train_qidxs
    train_dict['pidxs'] = train_pidxs

    val_qidxs, val_pidxs = [], []
    for i in range(len(val_dict['landmark_id'])):
        if val_dict['landmark_id'][i] != -1:
            if i != len(val_dict['landmark_id'])-1 and val_dict['landmark_id'][i] == val_dict['landmark_id'][i+1]:
                val_qidxs.append(i)
                val_pidxs.append(i+1)
    val_dict['qidxs'] = val_qidxs
    val_dict['pidxs'] = val_pidxs

    # warp all dict to one pickle file
    all_dict = {'train': train_dict, 'val': val_dict}
    pickle_file = open(pkl_output_path, 'wb')
    pickle.dump(all_dict, pickle_file)
    print(">> save {} done...".format(pkl_output_path.split('/')[-1]))

