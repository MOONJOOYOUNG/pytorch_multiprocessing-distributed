from model import resnet
import data as dataset
import utils
import plot_curves
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data
import torch.utils.data.distributed
import torch.multiprocessing as mp
import time
import shutil
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

parser = argparse.ArgumentParser(description='Confidence Aware Learning')
parser.add_argument('--batch_size', default= 64, type=int, help='Batch size')
parser.add_argument('--epochs', default= 20, type=int, help='Total number of epochs to run')
parser.add_argument('--model', default='res', type=str, help='Models name to use [res, dense, vgg]')
parser.add_argument('--save_path', default='./test/', type=str, help='Savefiles directory')
parser.add_argument('--gpu', default='7', type=str, help='GPU id to use')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--world_size', default=2, type=int, help='Gpu use number')

args = parser.parse_args()

def main(rank, world_size):
    init_process(rank, world_size)

    # make dataloader
    train_loader, test_loader = dataset.get_loader(args, rank, world_size)

    # set model
    if args.model == 'res':
        model = resnet.ResNet18()

    model = model.cuda(rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])
    cudnn.benchmark = True

    # set criterion
    criterion = nn.CrossEntropyLoss().cuda(rank)

    # set optimizer (default:sgd)
    optimizer = optim.SGD(model.parameters(),
                          lr=0.1,
                          momentum=0.9,
                          weight_decay=0.0001,
                          nesterov=True)
    # set scheduler
    scheduler = MultiStepLR(optimizer,
                            milestones=[60,80],
                            gamma=0.1)

    # make logger
    train_logger = utils.Logger(os.path.join(args.save_path, 'train.log'))
    test_logger = utils.Logger(os.path.join(args.save_path, 'test.log'))


    # Start Train
    for epoch in range(1, args.epochs + 1):
        # scheduler
        if dist.get_rank() == 0:
            scheduler.step()
        # Train
        train(train_loader, model, criterion, optimizer, epoch, train_logger)
        validate(test_loader, model, criterion, epoch, test_logger, 'test')
        # Save Model each epoch
        if dist.get_rank() == 0:
            if epoch == int(args.epochs):
                torch.save(model.state_dict(), os.path.join(args.save_path, '{0}_{1}.pth'.format('model', epoch)))
    # Finish Train

    # Draw Plot
    if dist.get_rank() == 0:
        plot_curves.draw_plot(args.save_path)

    dist.destroy_process_group()


def train(train_loader, model, criterion, optimizer, epoch, logger):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()


    end = time.time()

    model.train()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # set input ,target
        input, target = input.cuda(dist.get_rank()), target.cuda(dist.get_rank())

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec, correct = utils.accuracy(output, target)
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if dist.get_rank() == 0 :
            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'            
                      'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1))

    if dist.get_rank() == 0:
        logger.write([epoch, losses.avg, top1.avg])



def validate(val_loader, model, criterion, epoch, logger, mode):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    end = time.time()
    total_acc = 0

    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.cuda(dist.get_rank()), target.cuda(dist.get_rank())

            # compute output
            output = model(input)
            loss = criterion(output, target)

            pred = output.data.max(1, keepdim=True)[1]
            total_acc += pred.eq(target.data.view_as(pred)).sum()

            # measure accuracy and record loss
            prec, correct = utils.accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if dist.get_rank() == 0:
                if i % args.print_freq == 0:
                    print(mode, ': [{0}/{1}]\t'
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses))
        total_acc = 100. * total_acc / len(val_loader.dataset)
    if dist.get_rank() == 0:
        print('Accuracy {:.2f}'.format(total_acc))
        logger.write([epoch, losses.avg, total_acc.item()])

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def run_model(main, world_size):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    shutil.copy(__file__, os.path.join(args.save_path, 'main.py'))

    mp.spawn(main,
             args=(world_size,),
             nprocs=world_size,
             join=True)

def init_process(rank, world_size, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '20080'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

if __name__ == "__main__":
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    run_model(main, args.world_size)

