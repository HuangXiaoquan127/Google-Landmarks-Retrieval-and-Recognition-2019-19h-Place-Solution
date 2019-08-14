import os
import hashlib

def get_root():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))


def get_data_root():
    # hxq modified
    # return os.path.join(get_root(), 'data')

    # return '/home/iap205/Datasets/retrieval-SfM'
    return '/home/iap205/Datasets/google-landmarks-dataset-resize'
    # return '/media/iap205/Data960G/Datasets/GLD-v2'
    # return '/media/iap205/Data4T/Datasets/google-landmarks-dataset'
    # return '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2'

    # if dataset == 'SfM-120k':
    #     return '/home/iap205/Datasets/retrieval-SfM'
    # elif dataset == 'GLD-v1-resize':
    #     return '/home/iap205/Datasets/google-landmarks-dataset-resize'
    # elif dataset == 'GLD-v2':
    #     return '/media/iap205/Data4T/Datasets/google-landmarks-dataset-v2'
    # elif dataset == 'GLD-v1':
    #     return '/media/iap205/Data4T/Datasets/google-landmarks-dataset'


def htime(c):
    c = round(c)
    
    days = c // 86400
    hours = c // 3600 % 24
    minutes = c // 60 % 60
    seconds = c % 60

    if days > 0:
        return '{:d}d {:d}h {:d}m {:d}s'.format(days, hours, minutes, seconds)
    if hours > 0:
        return '{:d}h {:d}m {:d}s'.format(hours, minutes, seconds)
    if minutes > 0:
        return '{:d}m {:d}s'.format(minutes, seconds)
    return '{:d}s'.format(seconds)


def sha256_hash(filename, block_size=65536, length=8):
    sha256 = hashlib.sha256()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha256.update(block)
    return sha256.hexdigest()[:length-1]