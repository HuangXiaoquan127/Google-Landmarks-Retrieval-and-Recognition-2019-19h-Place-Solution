import pickle
import csv


pkl = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2/google-landmarks-dataset-v2.pkl'
save_path = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2/validation.csv'
with open(pkl, 'rb') as f:
    db = pickle.load(f)['val']

file = open(save_path, 'w')
writer = csv.writer(file)
for i in range(len(db['id'])):
    writer.writerow([db['id'][i], db['landmark_id'][i]])
file.close()
print('>> val from pkl to val done')