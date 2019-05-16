#!/usr/bin/python3
#
# Example run:
# ./main.py --model msdnet -b 2 -j 2 cifar10 --msd-blocks 10 --msd-base 4 --msd-step 2 \
#  --msd-stepmode even --growth 6-12-24 --gpu 0
# For evaluation / resume add: --resume --evaluate

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
import shutil
import time
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from utils import measure_model
from opts import args
from tools import write_file
from tools import write_csv
from buget_computation import dynamic_evaluate

# Init Torch/Cuda
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
best_prec1 = 0

acc_file_path = ""
acc_file_name = ""
flops_file_path = ""
flops_file_name = ""


def main(**kwargs):
    global args, best_prec1, acc_file_name, flops_file_name

    # Override if needed
    # kwargs is a dict
    for arg, v in kwargs.items():
        args.__setattr__(arg, v)  # set args.arg=v, copy kwargs to args

    ### Calculate FLOPs & Param
    model = getattr(models, args.model)(args)  ###### ? get models.(args.model)

    if args.data in ['cifar10', 'cifar100']:
        IMAGE_SIZE = 32
        if(args.data == 'cifar10'):
            acc_file_name = "cifar10_acc.csv"
            flops_file_name = "cifar10_flops.csv"
        else:
            acc_file_name = "cifar100_acc.csv"
            flops_file_name = "cifar100_flops.csv"
    else:
        IMAGE_SIZE = 224

    set_save_path()

    # calculate the FLOPs
    # if (args.evaluate):
    n_flops, n_params = measure_model(model, IMAGE_SIZE, IMAGE_SIZE,flops_file_path, args.debug)  #####

    if 'measure_only' in args and args.measure_only:  # no 'measure_only' parameter
        return n_flops, n_params

    print('Starting.. FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6))
    args.filename = "%s_%s_%s.txt" % (args.model, int(n_params), int(n_flops))
    # del(model)

    # Create model(the 2nd)
    # model = getattr(models, args.model)(args)  ####

    if args.debug:
        print(args)
        # print(model)

    model = torch.nn.DataParallel(model).cuda()  # Implements data parallelism at the module level
    # if exist mulit-devices, This container parallelizes the application of the given module by splitting
    # the input across the specified devices by chunking in the batch dimension (other objects will be copied once per device)

    ### Data loading, no nomarlisation
    if args.data == "cifar10":
        train_set = datasets.CIFAR10('../data', train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                     ]))
        val_set = datasets.CIFAR10('../data', train=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                   ]))
    elif args.data == "cifar100":
        train_set = datasets.CIFAR100('../data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                      ]))
        val_set = datasets.CIFAR100('../data', train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]))

    cudnn.benchmark = True  ######

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # test_loader = torch.utils.data.DataLoader(
    #     val_set,
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)
    test_loader = val_loader

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    ##### validate
    # Resume from a checkpoint
    if args.resume:  # use latest checkpoint if have any
        checkpoint = load_checkpoint(args)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    # Evaluate from a model
    if args.evaluate_from is not None:  # path to saved checkpoint
        args.evaluate = True
        state_dict = torch.load(args.evaluate_from)['state_dict']
        model.load_state_dict(state_dict)

        if args.evalmode is not None:
            if args.evalmode == 'anytime':
                validate(val_loader, model, criterion)
            else:
                print('dynamic')
                dynamic_evaluate(model, test_loader, val_loader, args)
            return

    # Run Forward / Backward passes
    # evaluate and return
    # if args.evaluate:
    #     validate(val_loader, model, criterion)
    #     return

    ###### train
    best_epoch=0
    for epoch in range(args.start_epoch, args.epochs):
        # Train for one epoch
        tr_prec1, tr_prec5, loss, lr = train(train_loader, model, criterion, optimizer, epoch)

        # Evaluate on validation set
        val_prec1, val_prec5 = validate(val_loader, model, criterion)

        # Remember best prec@1 and save checkpoint
        is_best = val_prec1 < best_prec1
        best_prec1 = max(val_prec1, best_prec1)
        if(is_best):
            best_epoch = epoch
        model_filename = 'checkpoint_%03d.pth.tar' % epoch
        save_checkpoint({
            'epoch': epoch,
            'model': args.model,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, args, is_best, model_filename, "%.4f %.4f %.4f %.4f %.4f %.4f\n" %
                                          (val_prec1, val_prec5, tr_prec1, tr_prec5, loss, lr))

    print('Best val_prec1: {:.4f} at epoch {}'.format(best_prec1, best_epoch))
    # TestModel and return
    model = model.cpu().module
    model = nn.DataParallel(model).cuda()
    # if args.debug:
    #     print(model)
    validate(val_loader, model, criterion)
    # n_flops, n_params = measure_model(model, IMAGE_SIZE, IMAGE_SIZE, flops_file_path,args.debug)
    # print('Finished training! FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6))
    print('Please run again with --resume --evaluate flags,'' to evaluate the best model.')

    return


def set_save_path():
    cwd = os.getcwd()
    fd_str = os.path.join(cwd, "output")
    if (not os.path.exists(fd_str)):
        os.mkdir(fd_str)
    global acc_file_path, flops_file_path, acc_file_name, flops_file_name
    acc_file_path = os.path.join(fd_str, acc_file_name)
    flops_file_path = os.path.join(fd_str, flops_file_name)
    # if(os.path.exists(acc_file_path)):
    #     os.remove(acc_file_path)
    # if(os.path.exists(flops_file_path)):
    #     os.remove(flops_file_path)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_per_cls = [AverageMeter() for i in range(0, model.module.num_blocks)]
    ###### initialize a AverageMeter() list whose size is the number of classifiers
    top5_per_cls = [AverageMeter() for i in range(0, model.module.num_blocks)]

    ### Switch to train mode
    model.train()

    running_lr = None

    end = time.time()
    # i: batch_index, target:
    for i, (input, target) in enumerate(train_loader):
        progress = float(epoch * len(train_loader) + i) / (args.epochs * len(train_loader))
        args.progress = progress

        ### Adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                  nBatch=len(train_loader), method=args.lr_type)
        if running_lr is None:
            running_lr = lr

        ### Measure data loading time
        data_time.update(time.time() - end)

        # target = target.cuda(async=True)
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        ### Compute output
        # with torch.no_grad():
        output = model(input_var, progress)
        if args.model == 'msdnet':
            loss = msd_loss(output, target_var, criterion)
        else:
            loss = criterion(output, target_var)

        ### Measure accuracy and record loss
        if hasattr(output, 'data'):  # output contains 'data'
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        elif args.model == 'msdnet':
            prec1, prec5, _ = msdnet_accuracy(output, target, input)
        else:
            raise NotImplementedError
        # losses.update(loss.data[0], input.size(0))
        losses.update(loss.data.item(), input.size(0))
        # top1.update(prec1[0], input.size(0))
        top1.update(prec1.item(), input.size(0))
        # top5.update(prec5[0], input.size(0))
        top5.update(prec5.item(), input.size(0))

        ### Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # _, _, (ttop1s, ttop5s) = msdnet_accuracy(output, target, input, val=True)
        ###### ttop1s and ttop5s: the list stores prec1 or prec5 for each classifier, whose size is the number of classifier
        # for c in range(0, len(top1_per_cls)):  ####### for each classifier
        #     top1_per_cls[c].update(ttop1s[c], input.size(0))  ####### top1_per_cls[c] stores prec1 for classifier c(1¬10) for all epoches
        #     top5_per_cls[c].update(ttop5s[c], input.size(0))


        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f}\t'  # ({batch_time.avg:.3f}) '
                  'Data {data_time.val:.3f}\t'  # ({data_time.avg:.3f}) '
                  'Loss {loss.val:.4f}\t'  # ({loss.avg:.4f}) '
                  'Prec@1 {top1.val:.3f}\t'  # ({top1.avg:.3f}) '
                  'Prec@5 {top5.val:.3f}\t'  # ({top5.avg:.3f})'
                  'lr {lr: .4f}'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=lr))

    # print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    # for c in range(0, len(top1_per_cls)):  ####### for each classifier
    #     print(' * For classifier {cls}: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
    #           .format(cls=c, top1=top1_per_cls[c], top5=top5_per_cls[c]))
    #     # save to the file
    #     print(acc_file_path)
    #     write_file(acc_file_path, str(top1_per_cls[c].avg))

    return 100. - top1.avg, 100. - top5.avg, losses.avg, running_lr


def msd_loss(output, target_var, criterion):
    losses = []
    for out in range(0, len(output)):
        losses.append(criterion(output[out], target_var))
    mean_loss = sum(losses) / len(output)
    return mean_loss


def msdnet_accuracy(output, target, x, val=False):
    """
    Calculates multi-classifier accuracy

    :param output: A list in the length of the number of classifiers,
                   including output tensors of size (batch, classes)
    :param target: a tensor of length batch_size, including GT
    :param x: network input
    :param val: A flag to print per class validation accuracy
    :return: mean precision of top1 and top5
    """

    top1s = []
    top5s = []
    prec1 = torch.FloatTensor([0]).cuda()
    prec5 = torch.FloatTensor([0]).cuda()
    for out in output:
        tprec1, tprec5 = accuracy(out.data, target, topk=(1, 5))
        prec1 += tprec1
        prec5 += tprec5
        # top1s.append(tprec1[0])
        top1s.append(tprec1.item())  # the list storing tprec1 for each classifer
        # top5s.append(tprec5[0])
        top5s.append(tprec5.item())

    # if val:
    #     for c in range(0, len(top1s)):
    #         print("Classifier {} top1: {} top5: {}".format(c, top1s[c], top5s[c]))
    prec1 = prec1 / len(output)  # mean precision of top1 for all classifiers
    prec5 = prec5 / len(output)  # mean precision of top5 for all classifiers
    return prec1, prec5, (top1s, top5s)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_per_cls = [AverageMeter() for i in range(0, model.module.num_blocks)]
    ###### initialize a AverageMeter() list whose size is the number of classifiers
    top5_per_cls = [AverageMeter() for i in range(0, model.module.num_blocks)]

    ### Switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # target = target.cuda(async=True)
        target = target.cuda(non_blocking=True)
        # input_var = torch.autograd.Variable(input, volatile=True)
        # target_var = torch.autograd.Variable(target, volatile=True)

        ### Compute output
        with torch.no_grad():
            output = model(input, 0.0)
            if args.model == 'msdnet':
                loss = msd_loss(output, target, criterion)  #####?
            else:
                loss = criterion(output, target)

        ### Measure accuracy and record loss
        if hasattr(output, 'data'):
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        elif args.model == 'msdnet':
            prec1, prec5, _ = msdnet_accuracy(output, target, input)
        # losses.update(loss.data[0], input.size(0))
        losses.update(loss.data.item(), input.size(0))
        # top1.update(prec1[0], input.size(0))
        top1.update(prec1.item(), input.size(0))  ####### a list storing mean precision of top1 for all classifiers
        # top5.update(prec5[0], input.size(0))
        top5.update(prec5.item(), input.size(0))

        ### Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     print('Test: [{0}/{1}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        #           'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        #               i, len(val_loader), batch_time=batch_time, loss=losses,
        #               top1=top1, top5=top5))

        _, _, (ttop1s, ttop5s) = msdnet_accuracy(output, target, input, val=True)
        ###### ttop1s and ttop5s: the list stores prec1 or prec5 for each classifier, whose size is the number of classifier
        for c in range(0, len(top1_per_cls)):  ####### for each classifier
            top1_per_cls[c].update(ttop1s[c], input.size(0))  ####### top1_per_cls[c] stores prec1 for classifier c(1¬10) for all epoches
            top5_per_cls[c].update(ttop5s[c], input.size(0))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    for c in range(0, len(top1_per_cls)):  ####### for each classifier
        print(' * For classifier {cls}: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(cls=c, top1=top1_per_cls[c], top5=top5_per_cls[c]))
        # save to the file
        print(acc_file_path)
        # write_file(acc_file_path, str(c)+','+str(top1_per_cls[c].avg))
        write_csv(acc_file_path,[str(c),str(top1_per_cls[c].avg)])

    return 100. - top1.avg, 100. - top5.avg  ######### the final clasifier's mean value


def load_checkpoint(args):
    if args.evaluate_from:
        print("Evaluating from model: ", args.evaluate_from)
        model_filename = args.evaluate_from
    else:
        model_dir = os.path.join(args.savedir, 'save_models')
        latest_filename = os.path.join(model_dir, 'latest.txt')
        if os.path.exists(latest_filename):
            with open(latest_filename, 'r') as fin:
                model_filename = fin.readlines()[0].strip()
        else:
            return None
    print("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    print("=> loaded checkpoint '{}'".format(model_filename))
    return state


def save_checkpoint(state, args, is_best, filename, result):
    # print(args)
    print('args.savedir', args.savedir)
    result_filename = os.path.join(args.savedir, args.filename)
    model_dir = os.path.join(args.savedir, 'save_models')
    model_filename = os.path.join(model_dir, filename)
    latest_filename = os.path.join(model_dir, 'latest.txt')
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)
        os.makedirs(model_dir)

    # For mkdir -p when using python3
    # os.makedirs(args.savedir, exist_ok=True)
    # os.makedirs(model_dir, exist_ok=True)

    print("=> saving checkpoint '{}'".format(model_filename))
    with open(result_filename, 'a') as fout:
        fout.write(result)
    torch.save(state, model_filename)
    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)        # write the current model filename
    if args.no_save_model:
        shutil.move(model_filename, best_filename)
    elif is_best:
        shutil.copyfile(model_filename, best_filename)

    print("=> saved checkpoint '{}'".format(model_filename))
    return


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args, batch=None,
                         nBatch=None, method='cosine'):
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        if args.data in ['cifar10', 'cifar100']:
            lr, decay_rate = args.lr, 0.1
            if epoch >= args.epochs * 0.75:
                lr *= decay_rate ** 2
            elif epoch >= args.epochs * 0.5:
                lr *= decay_rate
        else:
            """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
            lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()