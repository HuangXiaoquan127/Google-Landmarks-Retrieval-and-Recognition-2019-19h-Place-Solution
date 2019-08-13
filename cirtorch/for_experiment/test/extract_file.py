import shutil
import os
import csv


extract_table_path = '/home/iap205/Datasets/index_temp.csv'
extract_to = '/home/iap205/Datasets/index_temp'

file = open(extract_table_path, 'r')
file_reader = csv.reader(file)

i = 0
for path in file_reader:
    shutil.move(os.path.join('/home/iap205/Datasets/google-landmarks-dataset-test', path[0]+'.jpg'), extract_to)
    i += 1
print('>> extract {}'.format(i))
