import os
import csv


if __name__ == '__main__':
    img_path = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2/index'
    file_path = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2/index.csv'
    mark_path = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2/index_mark.csv'
    miss_path = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2/index_miss.csv'

    files = os.listdir(img_path)
    files = {files[i].split('.jpg')[0]: files[i].split('.jpg')[0] for i in range(len(files))}
    csvfile = open(file_path, 'r')
    csvreader = csv.reader(csvfile)
    savefile = open(mark_path, 'w')
    save_writer = csv.writer(savefile)
    for i, line in enumerate(csvreader):
        if i != 0:
            if files.__contains__(line[0]):
                save_writer.writerow(['1', line[0]])
                files.pop(line[0])
            else:
                save_writer.writerow(['0', line[0]])
    csvfile.close()
    savefile.close()
    print('>> done...')
