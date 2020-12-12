import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD, Adam
import numpy as np
import argparse
import time
from sklearn.metrics.cluster import normalized_mutual_info_score
from pathlib import Path
import itertools

from models.cnn.model import SeqCNN
import clustering
from src.Cohortney.data_utils import load_data, sep_hawkes_proc
from src.Cohortney.utils import *


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--maxsize', type=int, default=None)
    parser.add_argument('--nmb_cluster', type=int, default=10)
    parser.add_argument('--maxlen', type=int, default=2000)
    parser.add_argument('--ext', type=str, default='txt')
    parser.add_argument('--not_datetime', action='store_true')
    parser.add_argument('--gamma', type=float, default=1.4)
    parser.add_argument('--Tb', type=float, default=7e-6)
    parser.add_argument('--Th', type=float, default=80)
    parser.add_argument('--N', type=int, default=1500)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    ss, _, class2idx, user_list = load_data(Path(args.data_dir), maxsize=args.maxsize, maxlen=args.maxlen, ext=args.ext, datetime=not args.not_datetime)
    
    #weird
    events = itertools.chain.from_iterable([sep_hawkes_proc(user_list, i) for i in range(len(class2idx))])
    grid = make_grid(args.gamma, args.Tb, args.Th, args.N, args.n)

    T_j = grid[-1]
    Delta_T = np.linspace(0, grid[-1], 2**args.n)
    Delta_T = Delta_T[Delta_T< int(T_j)]
    Delta_T = tuple(Delta_T)

    array, _ = arr_func(events, T_j, Delta_T)

    #dataset = Dataset(array)
    dataset = [torch.FloatTensor(x) for x in array]
    if args.verbose:
        print('Loaded data')

    input_size = dataset[0].shape[0]

    model = SeqCNN(input_size)
    model.top_layer = None
    model.to(device)
    fd = model.fd
    
    # optimizer = SGD(
    #     filter(lambda x: x.requires_grad, model.parameters()),
    #     lr=args.lr,
    #     momentum=args.momentum,
    #     weight_decay=args.wd,
    # )

    optimizer = Adam(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.wd,
    )

    criterion = nn.CrossEntropyLoss()

    dataloader = torch.utils.data.DataLoader(dataset, 
                                             #collate_fn=pad_collate1,
                                             batch_size=args.batch,
                                             num_workers=args.workers,
                                             pin_memory=True)

    deepcluster = clustering.Kmeans(args.nmb_cluster)

    for epoch in range(args.start_epoch, args.epochs):
        end = time.time()

        # remove head
        model.top_layer = None
        #model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

        # get the features for the whole dataset
        features = compute_features(dataloader, model, len(dataset), device)

        # cluster the features
        if args.verbose:
            print('Cluster the features')
        clustering_loss = deepcluster.cluster(features, verbose=args.verbose)

        # assign pseudo-labels
        if args.verbose:
            print('Assign pseudo labels')
        train_dataset = clustering.cluster_assign(deepcluster.lists,
                                                  dataset)

        # uniformly sample per target
        # sampler = UnifLabelSampler(int(args.reassign * len(train_dataset)),
        #                            deepcluster.lists)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
        #    collate_fn=pad_collate2,
            batch_size=args.batch,
            num_workers=args.workers,
        #    sampler=sampler,
            pin_memory=True,
        )

        # set last fully connected layer
        # mlp = list(model.classifier.children())
        # mlp.append(nn.ReLU(inplace=True).to(device))
        # model.classifier = nn.Sequential(*mlp)
        model.top_layer = nn.Linear(fd, len(deepcluster.lists))
        model.top_layer.weight.data.normal_(0, 0.01)
        model.top_layer.bias.data.zero_()
        model.top_layer.to(device)

        # train network with clusters as pseudo-labels
        end = time.time()
        loss = train(train_dataloader, model, criterion, optimizer, epoch, device)

        # print log
        if args.verbose:
            print(f'###### Epoch {epoch} ###### \n Time: {(time.time() - end):.3f} s\n Clustering loss: {clustering_loss:.3f} \n ConvNet loss: {loss:.3f}')
            # try:
            #     nmi = normalized_mutual_info_score(
            #         clustering.arrange_clustering(deepcluster.lists),
            #         clustering.arrange_clustering(cluster_log.data[-1])
            #     )
            #     print('NMI against previous assignment: {0:.3f}'.format(nmi))
            # except IndexError:
            #     pass
            print('####################### \n')
        # # save running checkpoint
        # torch.save({'epoch': epoch + 1,
        #             'arch': args.arch,
        #             'state_dict': model.state_dict(),
        #             'optimizer' : optimizer.state_dict()},
        #            os.path.join(args.exp, 'checkpoint.pth.tar'))

        # save cluster assignments
        # cluster_log.log(deepcluster.lists)


def train(loader, model, crit, opt, epoch, device):
    """Training of the CNN.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """
    # batch_time = AverageMeter()
    # losses = AverageMeter()
    # data_time = AverageMeter()
    # forward_time = AverageMeter()
    # backward_time = AverageMeter()

    total_loss = 0
    N = 0

    # switch to train mode
    model.train()

    # create an optimizer for the last fc layer
    # optimizer_tl = SGD(
    #     model.top_layer.parameters(),
    #     lr=args.lr,
    #     weight_decay=10**args.wd,
    # )

    optimizer_tl = Adam(
        model.top_layer.parameters(),
        lr=args.lr,
        weight_decay=10**args.wd,
    )

    end = time.time()
    for i, (input_tensor, target) in enumerate(loader):
        #data_time.update(time.time() - end)

        # save checkpoint
        # n = len(loader) * epoch + i
        # if n % args.checkpoints == 0:
        #     path = os.path.join(
        #         args.exp,
        #         'checkpoints',
        #         'checkpoint_' + str(n / args.checkpoints) + '.pth.tar',
        #     )
        #     if args.verbose:
        #         print('Save checkpoint at: {0}'.format(path))
        #     torch.save({
        #         'epoch': epoch + 1,
        #         'arch': args.arch,
        #         'state_dict': model.state_dict(),
        #         'optimizer' : opt.state_dict()
        #     }, path)

        #target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input_tensor.to(device))
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = crit(output, target_var)

        total_loss += loss.item()
        N += input_tensor.shape[0]

        # record loss
        #losses.update(loss.data[0], input_tensor.size(0))

        # compute gradient and do SGD step
        opt.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        optimizer_tl.step()

        # measure elapsed time
        #batch_time.update(time.time() - end)
        end = time.time()

        # if args.verbose and (i % 200) == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss: {loss.val:.4f} ({loss.avg:.4f})'
        #           .format(epoch, i, len(loader), batch_time=batch_time,
        #                   data_time=data_time, loss=losses))

    avg_loss = total_loss / N

    return avg_loss

@torch.no_grad()
def compute_features(dataloader, model, N, device):
    if args.verbose:
        print('Compute features')
    #batch_time = AverageMeter()
    # end = time.time()
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor) in enumerate(dataloader):
        input_var = torch.autograd.Variable(input_tensor)
        aux = model(input_var).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * args.batch: (i + 1) * args.batch] = aux
        else:
            # special treatment for final batch
            features[i * args.batch:] = aux

        # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()

    #     if args.verbose and (i % 200) == 0:
    #         print('{0} / {1}\t'
    #               'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
    #               .format(i, len(dataloader), batch_time=batch_time))
    return features
    

if __name__ == "__main__":
    args = parse_arguments()
    if args.seed is not None:
        random_seed(args.seed)
    main(args)
