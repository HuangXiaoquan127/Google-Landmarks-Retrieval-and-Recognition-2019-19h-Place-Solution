# https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
# https://www.zhihu.com/question/67726969
import torch
import torch.nn as nn
import os
import time
from cirtorch.networks.imageretrievalnet import init_network
import torchvision.transforms as transforms
from cirtorch.datasets.traindataset import TuplesDataset
from cirtorch.datasets.datahelpers import collate_tuples
from cirtorch.layers.loss import ContrastiveLoss
from cirtorch.examples.train import train


class DataParallelModel(nn.Module):

    def __init__(self, D_in, D_out):
        super().__init__()
        self.block1 = nn.Linear(D_in, 1024)

        # wrap block2 in DataParallel
        self.block2 = nn.Linear(1024, 1024)
        self.block2 = nn.DataParallel(self.block2)

        self.block3 = nn.Linear(1024, D_out)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    N, D_in, H, D_out = 10000, 8048, 1024, 100
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)
    train_epoch = 200
    # x.cuda()
    # y.cuda()

    N, C, H, W = 10, 3, 256, 256
    D_out = 2048
    x = torch.randn(N, C, H, W)
    y = torch.randn(D_out, N)

    model_params = {}
    model_params['architecture'] = 'resnet101'
    model_params['pooling'] = 'gem'
    model_params['local_whitening'] = False
    model_params['regional'] = False
    model_params['whitening'] = False
    # model_params['mean'] = ...  # will use default
    # model_params['std'] = ...  # will use default
    model_params['pretrained'] = True
    model = init_network(model_params)

    # model = torch.nn.Sequential(
    #     DataParallelModel(D_in, H),
    #     DataParallelModel(H, H),
    #     torch.nn.Linear(H, D_out)
    # )
    model.cuda()

    criterion = ContrastiveLoss(margin=0.85).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # # Data loading code
    # normalize = transforms.Normalize(mean=model.meta['mean'], std=model.meta['std'])
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     normalize,
    # ])
    # train_dataset = TuplesDataset(
    #     name='retrieval-SfM-120k',
    #     mode='train',
    #     imsize=362,
    #     nnum=5,
    #     qsize=2000,
    #     poolsize=22000,
    #     transform=transform
    # )
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=5, shuffle=True,
    #     num_workers=8, pin_memory=True, sampler=None,
    #     drop_last=True, collate_fn=collate_tuples
    # )
    # if args.val:
    #     val_dataset = TuplesDataset(
    #         name=args.training_dataset,
    #         mode='val',
    #         imsize=args.image_size,
    #         nnum=args.neg_num,
    #         qsize=float('Inf'),
    #         poolsize=float('Inf'),
    #         transform=transform
    #     )
    #     val_loader = torch.utils.data.DataLoader(
    #         val_dataset, batch_size=args.batch_size, shuffle=False,
    #         num_workers=args.workers, pin_memory=True,
    #         drop_last=True, collate_fn=collate_tuples
    #     )

    print('>> start to train...')
    start_time = time.time()
    for t in range(train_epoch):
        y_pred = model(x.cuda())
        loss = torch.nn.functional.mse_loss(y_pred, y.cuda())
        # loss = train(train_loader, model, criterion, optimizer, 100)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print('\r>> epoch {}/{} done...'.format(t, train_epoch), end='')
    print('')
    duration_time = time.time() - start_time
    print('>> duration_time: {}'.format(duration_time))

