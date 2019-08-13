import numpy as np
import csv


a = np.arange(9).reshape(3, 3)
savefile = open('/home/iap205/Datasets/Google-Landmarks_Dataset_resize/aa.csv', 'w')
writer = csv.writer(savefile)
writer.writerow([1, 'adf'])
savefile.close()

csvfile = open('/home/iap205/Datasets/Google-Landmarks_Dataset_resize/aa.csv', 'r')
csvreader = csv.reader(csvfile)
b = [line[:2][0] for line in csvreader]
print(b)