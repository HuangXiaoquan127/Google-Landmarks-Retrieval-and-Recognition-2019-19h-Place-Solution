import os
import shutil


if __name__ == "__main__":

    input_directory = '/home/iap205/PycharmProjects/cnnimageretrieval-end2end-google_landmarks_retrieval/' \
                      'YOUR_EXPORT_DIR/record/temp_log/'
    output_directory = '/home/iap205/PycharmProjects/cnnimageretrieval-end2end-google_landmarks_retrieval/' \
                       'YOUR_EXPORT_DIR/record/temp_log/'
    filenames = ['train_R101_CA_FC_GL_256', 'train_R101_CA_M2_FC_GL_256',
                 'train_R101_CA_B_FC_GL_256', 'train_R101_CA_M2_B_FC_GL_256',
                 'train_R101_M2_FC_GL2_655', 'train_R101_LA_FC_GL_256', ]

    for filename in filenames:
        filepath = os.path.join(input_directory, filename + '.txt')
        output_path = os.path.join(output_directory, filename + '_simplify' + '.txt')
        f = open(filepath)
        lines = f.readlines()
        f_simplify = open(output_path, 'w')
        for i in range(len(lines)):
            if lines[i][:5] == '>>>> ':
                if i > 0 and i < len(lines)-1:
                    if (lines[i-1][:5] == '>>>> ' and lines[i+1][:5] == '>>>> ') or \
                       (lines[i-1] == '\n' and lines[i+1][:5] == '>>>> '):
                        continue
            elif lines[i] == '\n':
                continue

            f_simplify.write(lines[i])
        f_simplify.close()
        print('>>>> simplify done...')
