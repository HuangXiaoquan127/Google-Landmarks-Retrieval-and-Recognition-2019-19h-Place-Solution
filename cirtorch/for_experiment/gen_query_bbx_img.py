import os
from PIL import Image, ImageDraw
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.general import get_data_root


def gen_query_bbx_img():
    datasets = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k']
    for dataset in datasets:
        cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))
        qimages = [cfg['qim_fname'](cfg, i) for i in range(cfg['nq'])]
        bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]
        width = 5

        for i in range(len(qimages)):
            im = Image.open(qimages[i])
            draw = ImageDraw.Draw(im)
            (x0, y0, x1, y1) = bbxs[i]
            for j in range(width):
                draw.rectangle([x0-j, y0-j, x1+j, y1+j], outline='yellow')
            im.save('_bbx.jpg'.join(qimages[i].split('.jpg')))

        print("{} qurery_bbx generate ok".format(dataset))