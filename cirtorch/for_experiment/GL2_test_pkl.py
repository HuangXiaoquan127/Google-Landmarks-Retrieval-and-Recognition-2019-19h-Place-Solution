import pickle
import csv
import pandas as pd
import numpy as np
import os
import shutil


if __name__ == '__main__':
    data_table_path = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2/train.csv'
    pkl_output_path = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2/test/' \
                      'GL2-test/gnd_GL2-test.pkl'
    train_img_path = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2'
    test_img_copy_path = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2/test/GL2-test/jpg'

    print(">> load {} file...".format(data_table_path.split('/')[-1]))
    csvfile = open(data_table_path, 'r')
    csvreader = csv.reader(csvfile)
    id_url_class_list = [line[:3] for line in csvreader][1:]
    csvfile.close()
    train_temp = [[id_url_class_list[i][0], id_url_class_list[i][2]] for i in range(len(id_url_class_list))]

    shuffle = np.random.permutation(len(train_temp))
    train = []
    for i in shuffle:
        train.append(train_temp[i])

    train = pd.DataFrame(train, columns=['id', 'landmark_id'])
    train['landmark_id'] = train['landmark_id'].astype('int')
    # train = train.sort_values('landmark_id')
    train = train.to_numpy()

    select_class_num = 500
    shuffle = np.random.permutation(max(train[:, 1]))[:select_class_num]
    test = []
    for i in range(len(train)):
        if train[i][1] in shuffle:
            test.append(train[i])

    test = pd.DataFrame(test, columns=['id', 'landmark_id'])
    test = test.sort_values('landmark_id')
    test = test.to_numpy()

    print(">> copy test image from train set...")
    imlist = list(test[:, 0])
    img_path = [os.path.join(train_img_path, '/'.join(list(imlist[i])[:3]), imlist[i] + '.jpg')
                for i in range(len(imlist))]
    for i in range(len(imlist)):
        shutil.copyfile(img_path[i], os.path.join(test_img_copy_path, imlist[i] + '.jpg'))

    # generate qidx, qimlist and gnd
    qidx, qimlist = [0], [test[0][0]]
    # gnd = [{'ok': []}] * select_class_num
    gnd = []
    for i in range(select_class_num):
        gnd.append({'ok': []})
    class_cnt = 0
    for i in range(1, len(test)):
        if test[i][1] != test[i-1][1]:
            qidx.append(i)
            qimlist.append(test[i][0])
            class_cnt += 1
        else:
            gnd[class_cnt]['ok'].append(i)

    test_pkl = {'imlist': imlist, 'qidx': qidx, 'qimlist': qimlist, 'gnd': gnd}
    pickle_file = open(pkl_output_path, 'wb')
    pickle.dump(test_pkl, pickle_file)
    pickle_file.close()
    print(">> save {} done...".format(pkl_output_path.split('/')[-1]))


