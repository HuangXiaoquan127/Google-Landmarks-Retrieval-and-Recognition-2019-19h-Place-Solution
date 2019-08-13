import os, csv
import numpy as np
from cirtorch.utils.general import get_data_root

csvfile = open(os.path.join(get_data_root(), 'index_clear.csv'), 'r')
csvreader = csv.reader(csvfile)
images = [line[:1][0] for line in csvreader][:100]

csvfile = open(os.path.join(get_data_root(), 'test_clear.csv'), 'r')
csvreader = csv.reader(csvfile)
qimages = [line[:1][0] for line in csvreader][:55]

ranks = np.zeros((100, 55))

# save to csv file
print(">> save to csv file...")
submission_file = open(os.path.join(get_data_root(), 'submission.csv'), 'w')
writer = csv.writer(submission_file)
test_clear_flag_file = open(os.path.join(get_data_root(), 'test_clear_flag.csv'), 'r')
csvreader = csv.reader(test_clear_flag_file)
top_num = 100
cnt = 0
for index, line in enumerate(csvreader):
    if cnt == 55:
        break
    (flag, img_name) = line[:2]
    if flag == '1':
        select = []
        for i in range(top_num):
            select.append(images[int(ranks[i, cnt])].split('/')[-1].split('.jpg')[0])
        cnt += 1
        writer.writerow([img_name.split('/')[-1].split('.jpg')[0], ' '.join(select)])
    else:
        # random_list = random.sample(range(0, len(images)), top_num)
        random_list = np.random.choice(len(images), top_num, replace=False)
        select = []
        for i in range(top_num):
            select.append(images[random_list[i]].split('/')[-1].split('.jpg')[0])
        writer.writerow([img_name.split('/')[-1].split('.jpg')[0], ' '.join(select)])
    if index % 10 == 0 or index == len(qimages):
        print('\r>>>> {}/{} done...'.format(index, len(qimages)))
submission_file.close()
test_clear_flag_file.close()

print('>> done...')
