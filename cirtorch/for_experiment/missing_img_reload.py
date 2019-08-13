import csv
import os


if __name__ == '__main__':
    index_file_path = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2/index.csv'
    index_mark_path = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2/index_mark.csv'
    index_miss_path = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2/index_miss.csv'

    index_file = open(index_file_path, 'r')
    index_reader = csv.reader(index_file)
    key_url_list = [line[:2] for line in index_reader][1:]
    index_file.close()

    index_mark = open(index_mark_path, 'r')
    mark_reader = csv.reader(index_mark)
    index_miss = open(index_miss_path, 'w')
    miss_writer = csv.writer(index_miss)

    miss_writer.writerow(['id', 'url'])
    for i, line in enumerate(mark_reader):
        if line[0] == '0':
            miss_writer.writerow(key_url_list[i])

    index_mark.close()
    index_miss.close()
    print('>> done...')
