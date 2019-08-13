#%%
import scipy.io as sio

#%%
path = '/media/iap205/Data4T/DatasetsArchive/New College/NewCollegeGroundTruth.mat'
data = sio.loadmat(path)

#%%
path = '/media/iap205/Data4T/Projects/NetVLAD/datasets_mat/tokyo247.mat'
data = sio.loadmat(path)

#%%
path = '/media/iap205/Data4T/Projects/NetVLAD/output/tokyo247.mat'
data = sio.loadmat(path)

#%%
path = '/media/iap205/Data4T/Projects/NetVLAD/output/Pitts.mat'
data = sio.loadmat(path)
