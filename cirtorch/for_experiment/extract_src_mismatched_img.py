import os
import shutil


if __name__ == "__main__":

    filepath = '/media/iap205/Data/Export/cnnimageretrieval-pytorch/mismatched_image/' \
               '@media@iap205@Data@Export@cnnimageretrieval-pytorch@trained_network@retrieval-SfM-120k' \
               '_resnet101_gem_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362' \
               '@model_best.pth.tar/multiscale-[1]_whitening-None/oxford5k/mismatched_img_path.txt'
    output_path = '/media/iap205/Data/Export/cnnimageretrieval-pytorch/mismatched_image/' \
               '@media@iap205@Data@Export@cnnimageretrieval-pytorch@trained_network@retrieval-SfM-120k' \
               '_resnet101_gem_contrastive_m0.85_adam_lr5.0e-07_wd1.0e-04_nnum5_qsize2000_psize22000_bsize5_imsize362' \
               '@model_best.pth.tar/multiscale-[1]_whitening-None/oxford5k'

    output_path = os.path.join(output_path, 'mismatched_source_img')
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    f = open(filepath)
    line = f.readline()
    while line:
        if list(line)[0] == 'q':
            query_num = '_'.join(line.split(' ')[0 : -1])
            query_num_path = os.path.join(output_path, query_num)
            print('\r>>>> Extract {} mismatched image done...'.format(query_num), end='')
            if not os.path.isdir(query_num_path):
                os.mkdir(query_num_path)
            img_name = line.split(' ')[-1].split('/')[-1].strip()
            shutil.copyfile(line.split(' ')[-1].strip(), os.path.join(query_num_path, '{}_'.format(query_num) + img_name))
            line = f.readline()
            line = f.readline()

            if list(line)[0] == 'r':
                fp = '_'.join(line.split(' ')[2: -1])
                fp_path = os.path.join(query_num_path, fp)
                if not os.path.isdir(fp_path):
                    os.mkdir(fp_path)
                line = f.readline()
                while list(line)[0] <='9' and list(line)[0] >= '0':
                    img_name = line.split(' ')[-1].split('/')[-1].strip()
                    rank = line.split(' ')[0]
                    shutil.copyfile(line.split(' ')[-1].strip(), os.path.join(fp_path, '{}_'.format(rank) + img_name))
                    line = f.readline()

            line = f.readline()
            line = f.readline()
            line = f.readline()

            if list(line)[0] == 'r':
                fn = '_'.join(line.split(' ')[2: -1])
                fn_path = os.path.join(query_num_path, fn)
                if not os.path.isdir(fn_path):
                    os.mkdir(fn_path)
                line = f.readline()
                while line and list(line)[0] <='9' and list(line)[0] >= '0':
                    img_name = line.split(' ')[-1].split('/')[-1].strip()
                    rank = line.split(' ')[0]
                    shutil.copyfile(line.split(' ')[-1].strip(), os.path.join(fn_path, '{}_'.format(rank) + img_name))
                    line = f.readline()

        line = f.readline()
