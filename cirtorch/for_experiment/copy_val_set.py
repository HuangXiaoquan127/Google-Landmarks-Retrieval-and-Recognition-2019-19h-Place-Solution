import os
import pickle
import shutil


if __name__ == '__main__':
    src = '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2'
    dst = '/home/iap205/Datasets/google-landmarks-dataset-v2-val20000'
    db_fn = os.path.join(src, 'google-landmarks-dataset-v2.pkl')
    with open(db_fn, 'rb') as f:
        db = pickle.load(f)['val']
    for i in range(len(db['id'])):
        shutil.copy(os.path.join(src, '/'.join(list(db['id'][i])[:3]), db['id'][i] + '.jpg'),
                    os.path.join(dst, db['id'][i] + '.jpg'))
        if i+1 % 100 == 0 or i == len(db['id'])-1:
            print('\r>>>> copy {}/{} done...'.format(i+1, len(db['id'])), end='')
    print('')
    print('>> copy val done...')
