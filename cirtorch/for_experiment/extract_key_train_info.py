import os
import shutil


if __name__ == "__main__":

    filepath = '/home/iap205/PycharmProjects/cnnimageretrieval-end2end-google_landmarks_retrieval/' \
               'YOUR_EXPORT_DIR/record/train_record/train_R101_O_GL_FC_overlap_multigem.txt'
    output_path = '/home/iap205/PycharmProjects/cnnimageretrieval-end2end-google_landmarks_retrieval/' \
                  'YOUR_EXPORT_DIR/record/train_record/train_R101_O_GL_FC_overlap_multigem_key_info.txt'

    f = open(filepath)
    lines = f.readlines()
    f_key_info = open(output_path, 'w')
    epoch = 1
    for i in range(len(lines)):
        if lines[i].startswith('>> Train: [{}][400/400]'.format(epoch))\
                or lines[i].startswith('>> google-landmarks-dataset-resize-test: mAP')\
                or lines[i].startswith('>> google-landmarks-dataset-resize-test: mP@k'):
            f_key_info.write(lines[i])
            if lines[i].startswith('>> Train: [{}][400/400]'.format(epoch)):
                epoch += 1
    f_key_info.close()
    print('>>>> extract key train info done...')
