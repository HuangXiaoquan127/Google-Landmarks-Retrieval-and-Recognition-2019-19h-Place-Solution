import os, csv
from cirtorch.datasets.datahelpers import clear_no_exist

'''
def ParseData(data_file):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [line[:2] for line in csvreader]
    return key_url_list[1:]  # Chop off header


def clear_no_exist(input_path, output_path, img_check_path):
    print(">> load {} file...".format(input_path.split('/')[-1]))
    key_url_list = ParseData(input_path)
    images = [os.path.join(img_check_path, key_url_list[i][0]) + '.jpg' for i in range(len(key_url_list))]

    print(">> clear {} file...".format(input_path.split('/')[-1]))
    images_clean = []
    savefile = open(output_path, 'w')
    save_writer = csv.writer(savefile)
    for i in range(len(images)):
        if os.path.isfile(images[i]):
            save_writer.writerow([1, key_url_list[i][0]])
            images_clean.append(images[i])
        else:
            save_writer.writerow([0, key_url_list[i][0]])

    savefile.close()
    print(">> save {} done...".format(output_path.split('/')[-1]))

    return images_clean
'''

if __name__ == "__main__":
    clear_no_exist('/home/iap205/Datasets/Google-Landmarks_Dataset_resize/index.csv',
                   '/home/iap205/Datasets/Google-Landmarks_Dataset_resize/index_clear.csv',
                   '/home/iap205/Datasets/Google-Landmarks_Dataset_resize/index_clear_flag.csv',
                   '/home/iap205/Datasets/Google-Landmarks_Dataset_resize/resize_index_image')
    clear_no_exist('/home/iap205/Datasets/Google-Landmarks_Dataset_resize/test.csv',
                   '/home/iap205/Datasets/Google-Landmarks_Dataset_resize/test_clear.csv',
                   '/home/iap205/Datasets/Google-Landmarks_Dataset_resize/test_clear_flag.csv',
                   '/home/iap205/Datasets/Google-Landmarks_Dataset_resize/resize_test_image')
    # clear_no_exist(file_path, output_path, check_path)


