import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# network_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/R101_O_GL_FC_multigem/google-landmarks-dataset-resize_resnet101_multigem_whiten_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362/model_epoch100.pth.tar'
# network_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/R101_O_120k_FC_multigem/retrieval-SfM-120k_resnet101_multigem_whiten_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize650/model_epoch101.pth.tar'
# network_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/R101_M2_FC_GL2_655/google-landmarks-dataset-v2_resnet101_multigem_whiten_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize655/model_epoch38.pth.tar'
# network_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/R101_CA_M2_FC_GL2_655/google-landmarks-dataset-v2_resnet101_multigem_whiten_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize655/model_epoch30.pth.tar'
# network_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/R101_CA_M2_FC_GL_256/google-landmarks-dataset-resize_resnet101_multigem_whiten_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize256/model_epoch20.pth.tar'
# network_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/R101_CA_M2_B_FC_GL_256/google-landmarks-dataset-resize_resnet101_boostgem_whiten_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize256/model_epoch13.pth.tar'
network_path = '/media/iap205/Data/Export/cnnimageretrieval-google_landmark_retrieval/trained_network/R101_O_GL_FC/google-landmarks-dataset-resize_resnet101_gem_whiten_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362/model_epoch114.pth.tar'

checkpoint = torch.load(network_path)
checkpoint['meta']['multi_layer_cat'] = 1
# checkpoint['meta']['pooling'] = 'gem'
torch.save(checkpoint, network_path)
print('>> resave done')