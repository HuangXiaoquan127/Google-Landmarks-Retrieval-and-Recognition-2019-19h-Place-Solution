#%%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime
import numpy as np

#%%
# resnet101 + GeM + contrastive
GLD2_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/train_cleaned_test/R101_GLD2_362/google-landmarks-dataset-v2_resnet101_gem_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362/2019-07-01 15:25:37.209345_log_merge.csv'
GLD2_cln_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/train_cleaned_test/R101_GLD2_cleaned_362/GLD-v2-cleaned_resnet101_gem_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362/2019-07-01 18:38:13.006021_log_merge.csv'
GLD2_cln_m2_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/train_cleaned_test/R101_GLD2_cleaned_m2_362/GLD-v2-cleaned-m2_resnet101_gem_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362/2019-07-01 14:44:26.294624_log_merge.csv'
GLD1_resize_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/train_cleaned_test/R101_GLD1_resize_362/google-landmarks-dataset-resize_resnet101_gem_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362/2019-07-01 18:30:01.857093_log_merge.csv'
SfM_120k_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/train_cleaned_test/R101_120k_362/retrieval-SfM-120k_resnet101_gem_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362/2019-07-01 18:30:17.300446_log_merge.csv'

# resnet101 + SPoC + FC512 + contrastive
GLD2_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/train_cleaned_test/R101_SPoC_FC512_GLD2_362/google-landmarks-dataset-v2_resnet101_spoc_whiten_contrastive_m0.85_adam_lr5.0e-07_wd0.0e+00_nnum5_qsize2000_psize22000_bsize5_imsize362/results_log.csv'
GLD2_cln_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/train_cleaned_test/R101_SPoC_FC512_GLD2_cleaned_362/GLD-v2-cleaned_resnet101_spoc_whiten_contrastive_m0.85_adam_lr5.0e-07_wd0.0e+00_nnum5_qsize2000_psize22000_bsize5_imsize362/results_log.csv'
GLD2_cln_m2_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/train_cleaned_test/R101_SPoC_FC512_GLD2_cleaned_m2_362/GLD-v2-cleaned-m2_resnet101_spoc_whiten_contrastive_m0.85_adam_lr5.0e-07_wd0.0e+00_nnum5_qsize2000_psize22000_bsize5_imsize362/results_log.csv'
GLD1_resize_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/train_cleaned_test/R101_SPoC_FC512_GLD1_resize_362/google-landmarks-dataset-resize_resnet101_spoc_whiten_contrastive_m0.85_adam_lr5.0e-07_wd0.0e+00_nnum5_qsize2000_psize22000_bsize5_imsize362/results_log.csv'
SfM_120k_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/train_cleaned_test/R101_SPoC_FC512_120k_362/retrieval-SfM-120k_resnet101_spoc_whiten_contrastive_m0.85_adam_lr5.0e-07_wd0.0e+00_nnum5_qsize2000_psize22000_bsize5_imsize362/results_log.csv'

GLD2 = pd.read_csv(GLD2_path)
GLD2_cln = pd.read_csv(GLD2_cln_path)
GLD2_cln_m2 = pd.read_csv(GLD2_cln_m2_path)
GLD1_resize = pd.read_csv(GLD1_resize_path)
SfM_120k = pd.read_csv(SfM_120k_path)


#%%
# plot the data
titles = ['Google landmarks dataset v2 val_test set', 'ROxford Medium', 'RParis Medium']
datas = ['GLD2_mAP', 'RO_M_mAP', 'RP_M_mAP']
for i in range(len(titles)):
    plt.close('all')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig = plt.figure()
    plt.title(titles[i])
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(GLD2['epochs'], GLD2['{}@100'.format(datas[i])], label='GLD-v2', linewidth=1.0, linestyle='-')
    ax.plot(GLD2_cln['epochs'], GLD2_cln['{}@100'.format(datas[i])], label='GLD-v2-cleaned', linewidth=1.0, linestyle='-.')
    ax.plot(GLD2_cln_m2['epochs'], GLD2_cln_m2['{}@100'.format(datas[i])], label='GLD-v2-cleaned2', linewidth=1.0, linestyle='--')
    ax.plot(GLD1_resize['epochs'], GLD1_resize['{}@100'.format(datas[i])], label='GLD-v1', linewidth=1.0, linestyle='-')
    ax.plot(SfM_120k['epochs'], SfM_120k['{}@100'.format(datas[i])], label='SfM-120k', linewidth=1.0, linestyle='-')
    ax.legend(loc='lower right')
    # ax.legend(loc='upper left')
    ax.set(xlabel="epochs", ylabel="mAP@100(%)")
    ax.grid(axis='y', b=True, which='both', fillstyle='none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(f'/home/iap205/Pictures/{datetime.datetime.now()}.eps', format='eps')
    plt.show()


#%%
# # plot the data for test
# df3 = pd.DataFrame(np.random.randn(100, 3), columns=['B', 'C', 'D']).cumsum()
# df3['A'] = pd.Series(list(range(len(df3))))
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(df3['A'], df3['B'], label='newa', linewidth=1.0, linestyle='-.')
# ax.plot(df3['A'], df3['C'], label='newb', linewidth=1.0, linestyle='--')
# ax.plot(df3['A'], df3['D'], label='newc', linewidth=1.0, linestyle='-')
# ax.set(xlabel="epochs", ylabel="mAP@100(%)")
# ax.grid(axis='y', b=True, which='both', fillstyle='none')
# ax.legend(loc='upper left')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.show()

#%%

