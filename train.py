import argparse
import os
import sys
import random
import shutil
import time
import warnings
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import datetime

from tensorboardX import SummaryWriter
import numpy as np
from loader import dataloader_cifar10, dataloader_cifar100, dataloader_imagenet
from utils import utils
from models import cifar10 as cifar_models
from models import imagenet as imagenet_models

from models.imagenet.resnet_bi_imagenet_set_2 import HardBinaryConv_react
from models.imagenet.resnet_bi_imagenet_set_2_2 import HardBinaryConv
from models.bin_module.binarized_modules import HardBinaryConv_cifar
from kurtosis import KurtosisWeight, RidgeRegularization, WeightRegularization
from functools import reduce
from utils import KD_loss

# import models as cifar10_models
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
os.environ['TORCH_HOME'] = 'models'
# writer = None
# Logger handle
import torchvision.models as models

TORCH_MODEL_NAMES = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

CIFAR10_MODEL_NAMES = sorted(name for name in cifar_models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(cifar_models.__dict__[name]))

IMAGENET_MODEL_NAMES = sorted(name for name in imagenet_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(imagenet_models.__dict__[name]))

COSTUME_MODEL_NAMES = sorted(map(lambda s: s.lower(),
                            set(IMAGENET_MODEL_NAMES + CIFAR10_MODEL_NAMES)))

ALL_MODEL_NAMES = sorted(map(lambda s: s.lower(),
                            set(COSTUME_MODEL_NAMES + TORCH_MODEL_NAMES)))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=ALL_MODEL_NAMES,
                    help='model architecture: ' +
                        ' | '.join(ALL_MODEL_NAMES) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 50<-10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default="tcp://127.0.0.1:23456", type=str,
                    help='url used to set up distributed training')
parser.add_argument('--master-addr', default="132.68.39.200", type=str,
                    help='address used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--log_path', default='log', type=str,
                    help='log for tensorboardX.')

parser.add_argument('--custom_resnet', dest='custom_resnet', action='store_true',
                    help='use custom_resnet model')
parser.add_argument('--reset_resume',  dest='reset_resume', action='store_true',
                    help='reset resume parameters')
parser.add_argument('--ede', action='store_true',
                    help='use ede backprop')
parser.add_argument('--w-kurtosis-target', type=float, default=1.8,
                    help='weight kurtosis value')
parser.add_argument('--w-lambda-kurtosis', type=float, default=1.0,
                    help='lambda for kurtosis regularization in the Loss')
parser.add_argument('--w-kurtosis', action='store_true',
                    help='use kurtosis for weights regularization')
parser.add_argument('--weight-name', nargs='+', type=str,
                    help='param name to add kurtosis loss')
parser.add_argument('--remove-weight-name', nargs='+', type=str,
                    help='layer name to remove from kurtosis loss')
parser.add_argument('--kurtosis-mode', dest='kurtosis_mode', default='avg', choices=['max', 'sum', 'avg'],
                    type=lambda s: s.lower(), help='kurtosis regularization mode')
parser.add_argument('--diffkurt', action='store_true', default=False,
                    help='train with different kurtosis target for each layer')
parser.add_argument('--kurtepoch', type=int, metavar='N', default=0,
                    help='train with kurtosis starting at epoch ')
parser.add_argument('--twoblock', action='store_true', default=False,
                    help='2 different type of blocks')
parser.add_argument('--dataset', dest='dataset', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'],
                    type=lambda s: s.lower(), help='dataset')
parser.add_argument('--imagenet_setting', action='store_true',
                    help='use imagenet setting_step_1')
parser.add_argument('--imagenet_setting_step_1', action='store_true',
                    help='use imagenet setting')
parser.add_argument('--imagenet_setting_step_2', action='store_true',
                    help='use imagenet setting')
parser.add_argument('--imagenet_setting_step_2_ts', action='store_true',
                    help='use imagenet setting')
parser.add_argument('-a_teacher', '--arch_teacher', metavar='ARCH_T', default='resnet18',
                    choices=ALL_MODEL_NAMES,
                    help='model architecture: ' +
                        ' | '.join(ALL_MODEL_NAMES) +
                        ' (default: resnet18)')
parser.add_argument('--custom_resnet_teacher', dest='custom_resnet_teacher', action='store_true',
                    help='use custom_resnet model')
parser.add_argument('--resume_teacher', default='', type=str, metavar='PATH',
                    help='path to teacher (default: none)')

# knoeledge distilation
parser.add_argument('--kd', action='store_true',help='use kd')
parser.add_argument('--react', action='store_true',help='use react training')
parser.add_argument('--alpha', default=0.9, type=float, help='weight for KD (Hinton)')
parser.add_argument('--temperature', default=4, type=float)
parser.add_argument('--beta', default=200, type=float)
parser.add_argument('--qk_dim', default=128, type=int)
best_acc1 = 0
best_epoch = 0
msglogger = logging.getLogger()

def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # log
    datatime_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.log_path = os.path.join(args.log_path, str(args.w_kurtosis_target),datatime_str)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global best_epoch
    global writer
    # global msglogger
    writer = SummaryWriter(args.log_path)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.log_path + '/log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    msglogger = logging.getLogger()
    args.gpu = gpu
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    msglogger.info('log dir is:{} '.format(args.log_path))
    msglogger.info(args)
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        os.environ['MASTER_PORT'] = '29500'
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = gpu
            os.environ['MASTER_PORT'] = '2950' + str(gpu)
        os.environ['MASTER_ADDR'] = args.master_addr

        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    if args.imagenet_setting_step_2_ts:
        if args.dataset == 'imagenet':
            if args.custom_resnet_teacher:
                model_teacher = imagenet_models.__dict__[args.arch_teacher](pretrained=True)
            else:
                model_teacher = models.__dict__[args.arch_teacher](pretrained=True)
        else:
            model_teacher = cifar_models.__dict__[args.arch_teacher]()
        model_teacher = nn.DataParallel(model_teacher, device_ids=[args.gpu]).cuda()
        if args.resume_teacher:
            if os.path.isfile(args.resume):
                msglogger.info("=> loading checkpoint for teacher'{}'".format(args.resume_teacher))
                if args.gpu is None:
                    checkpoint = torch.load(args.resume_teacher)
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = 'cuda:{}'.format(args.gpu)
                    checkpoint = torch.load(args.resume_teacher, map_location=loc)

                model_teacher.load_state_dict(checkpoint['state_dict'])
                msglogger.info("=> loaded checkpoint teacher'{}' (epoch {}) acc {}"
                      .format(args.resume_teacher, checkpoint['epoch'], best_acc1))
            else:
                msglogger.info("=> no checkpoint found at '{}'".format(args.resume_teacher))

        for p in model_teacher.parameters():
            p.requires_grad = False
        model_teacher.eval()

    # create model
    if args.custom_resnet:
        msglogger.info("=> using resnet18 custom model '{}'".format(args.arch))
        if args.dataset == 'cifar10':
            model = cifar_models.__dict__[args.arch]()
        else:
            model = imagenet_models.__dict__[args.arch](args.pretrained)
    else:
        msglogger.info("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=args.pretrained)

    msglogger.info(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = torch.nn.DataParallel(model, device_ids=[args.gpu]).cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    if args.dataset == 'imagenet':
        all_parameters = model.parameters()
        weight_parameters = []
        for pname, p in model.named_parameters():
            if p.ndimension() == 4 or 'conv' in pname:
                weight_parameters.append(p)
        weight_parameters_id = list(map(id, weight_parameters))
        other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

        optimizer = torch.optim.Adam(
            [{'params': other_parameters},
             {'params': weight_parameters, 'weight_decay': args.weight_decay}],
            lr=args.lr, )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (1.0 - step / args.epochs), last_epoch=-1)


    if args.imagenet_setting_step_2_ts:
        criterion_kl = KD_loss.DistributionLoss_layer().cuda(args.gpu)
    if args.react:
        criterion_kl = None
    criterion_kl_c = KD_loss.DistributionLoss().cuda(args.gpu)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            msglogger.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            if not args.reset_resume:
                args.start_epoch = checkpoint['epoch']
                best_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
                if args.gpu is not None:
                    best_acc1 = best_acc1.to(args.gpu)

            model.load_state_dict(checkpoint['state_dict'])
            msglogger.info("=> loaded checkpoint '{}' (epoch {}) acc {}"
                .format(args.resume, args.start_epoch, best_acc1))
        else:
            msglogger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if args.dataset == "imagenet":
        train_loader = dataloader_imagenet(split='train', batch_size=args.batch_size, data_path=args.data, distributed=False)
        val_loader = dataloader_imagenet(split='test', batch_size=args.batch_size, data_path=args.data, distributed=False)
    elif args.dataset == "cifar10":
        train_loader = dataloader_cifar10(split='train', batch_size=args.batch_size)
        val_loader = dataloader_cifar10(split='test', batch_size=args.batch_size)
    else:
        train_loader = dataloader_cifar100(split='train', batch_size=args.batch_size)
        val_loader = dataloader_cifar100(split='test', batch_size=args.batch_size)


    if args.evaluate:
        acc = validate(val_loader, model, criterion, args, 0)
        msglogger.info('test acc : {}'.format(acc))
        return

    # Kurtosis regularization on weights tensors
    weight_to_hook = {}
    if args.w_kurtosis:
        if args.weight_name[0] == 'all':
            all_convs = [n + '.weight' for n, m in model.named_modules() if
                         (isinstance(m, nn.Conv2d) or isinstance(m,HardBinaryConv_react) or isinstance(m,HardBinaryConv) or isinstance(m,HardBinaryConv_cifar))]
            weight_name = all_convs[1:]
            if args.remove_weight_name:
                for name in weight_name:
                    if args.remove_weight_name[0] in name:
                        weight_name.remove(name)
            msglogger.info("weight_name : {}, remove {}".format(weight_name, args.remove_weight_name))
        else:
            weight_name = args.weight_name
        for name in weight_name:
            curr_param = utils.find_weight_tensor_by_name(model, name)
            if curr_param is None:
                name = name.replace("weight", 'float_weight')  # QAT name
                curr_param = utils.find_weight_tensor_by_name(model, name)
            weight_to_hook[name] = curr_param

    for epoch in range(args.start_epoch, args.epochs):
        if args.ede:
            # * compute t/k in back-propagation
            t, k = utils.cpt_tk(epoch,args.epochs)
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) :
                    module.k = k.cuda()
                    module.t = t.cuda()

        if args.imagenet_setting_step_2_ts:
            train_teacher_student(train_loader, model, model_teacher, criterion, criterion_kl, criterion_kl_c, optimizer, epoch, args, weight_to_hook)
        else:
            train(train_loader, model, criterion, optimizer, epoch, args, weight_to_hook)

        acc1 = validate(val_loader, model, criterion, args, epoch)
        scheduler.step()
        msglogger.info('----LR----- {}'.format(scheduler.get_lr()))
        is_best = acc1 > best_acc1
        if is_best:
            best_epoch = epoch
        best_acc1 = max(acc1, best_acc1)
        writer.add_scalar("Best val Acc1", best_acc1, epoch)
        msglogger.info(' ***** Best acc is Acc@1 {}, epoch {}, log {}'.format(best_acc1, best_epoch, args.log_path))
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, save_path=args.log_path)

def train(train_loader, model, criterion, optimizer, epoch, args, weight_to_hook=None):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    losses_kurt = utils.AverageMeter('Loss_kurt', ':.4e')
    losses_ce = utils.AverageMeter('Loss_ce', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    progress = utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, losses_kurt, losses_ce,top1, top5],
        msglogger, prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        ############ kurt ###########
        hookF_weights = {}
        hookF_weights_L2 = {}
        hookF_weights_wr = {}
        idx = 0
        if args.diffkurt:
            if args.dataset == 'imagenet':
                args.w_kurtosis_target = [1.8, 1.4, 1.4, 1.4,
                                          1.4, 1.2, 1.4, 1.2, 1.2,
                                          1.4, 1.4, 1.4, 1.2, 1.2,
                                          1.2, 1.2, 1.4, 1, 1]
            else:
                args.w_kurtosis_target = [1.4, 1.4, 1.4, 1.4,
                                          1.4, 1.4, 1.4, 1.4, 1.4,
                                          1.4, 1.4, 1.4, 1.4, 1.4,
                                          1.8, 1.8, 1.8, 1.8, 2.2]
        else:
            args.w_kurtosis_target = [args.w_kurtosis_target] * len(weight_to_hook)
        for name, w_tensor in weight_to_hook.items():
            hookF_weights[name] = KurtosisWeight(w_tensor, name, kurtosis_target=args.w_kurtosis_target[idx], k_mode=args.kurtosis_mode)
            if args.w_l2_reg:
                hookF_weights_L2[name] = RidgeRegularization(w_tensor, name)
            if args.w_wr_reg:
                hookF_weights_wr[name] = WeightRegularization(w_tensor, name)
            idx=idx+1
        ############ kurt ###########

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        images = images.cuda()
        output = model(images)
        orig_loss = criterion(output, target)
        ############ kurt loss ###########
        w_k_scale = 0
        w_kurtosis_regularization = 0
        if args.w_kurtosis and args.kurtepoch <= epoch:
            w_temp_values = []
            w_temp_kld_values = []
            w_kurtosis_loss = 0
            for w_kurt_inst in hookF_weights.values():
                w_kurt_inst.fn_regularization()
                w_temp_values.append(w_kurt_inst.kurtosis_loss)
                w_temp_kld_values.append(w_kurt_inst.KLDiv_loss)
            if args.kurtosis_mode == 'sum':
                w_kurtosis_loss = reduce((lambda a, b: a + b), w_temp_values)
            elif args.kurtosis_mode == 'avg':
                w_kurtosis_loss = reduce((lambda a, b: a + b), w_temp_values)
                w_kurtosis_loss = w_kurtosis_loss / len(weight_to_hook)
            elif args.kurtosis_mode == 'max':
                w_kurtosis_loss = reduce((lambda a, b: max(a, b)), w_temp_values)
            w_kurtosis_regularization = (10 ** w_k_scale) * args.w_lambda_kurtosis * w_kurtosis_loss
        ############ kurt loss ###########

        loss = orig_loss + w_kurtosis_regularization

        # measure accuracy and record loss
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        if args.w_kurtosis:
            losses_kurt.update(w_kurtosis_regularization.item(), images.size(0))
        losses_ce.update(orig_loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            # msglogger.info(progress.display(i))
            remain_epoch = args.epochs - epoch
            remain_iters = remain_epoch * len(train_loader) + (len(train_loader) - i)
            remain_seconds = remain_iters * batch_time.get_avg()
            seconds = (remain_seconds//1) % 60
            minutes = (remain_seconds//(1*60)) % 60
            hours = (remain_seconds//(1*60*60)) % 24
            days = (remain_seconds//(1*60*60*24))
            time_stamp = ""
            if (days > 0): time_stamp += "{} days, ".format(days)
            if (hours > 0) : time_stamp += "{} hr, ".format(hours)
            if (minutes > 0) : time_stamp += "{} min, ".format(minutes)
            if (seconds > 0) : time_stamp += "{} sec, ".format(seconds)            
            msglogger.info(">>>>>>>>>>>> Remaining Times: {}  <<<<<<<<<<<<<<<<<".format(time_stamp) )

    writer.add_scalar("Train Loss", loss.item(), epoch)
    writer.add_scalar("Train Acc1", top1.avg, epoch)
    writer.add_scalar("Train Acc5", top5.avg, epoch)

def train_teacher_student(train_loader, model_stud, model_teacher, criterion, criterion_kl, criterion_kl_c, optimizer, epoch, args,weight_to_hook=None):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    losses_kl = utils.AverageMeter('Loss_kl', ':.4e')
    losses_kl_c = utils.AverageMeter('Loss_kl_c', ':.4e')
    losses_kurt = utils.AverageMeter('Loss_kurt', ':.4e')
    losses_ce = utils.AverageMeter('Loss_ce', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    progress = utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, losses_kl, losses_kl_c, losses_kurt, losses_ce, top1, top5],
        msglogger, prefix="Epoch: [{}]".format(epoch))

    model_stud.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        images = images.cuda()

        ############ kurt ###########
        hookF_weights = {}
        hookF_weights_L2 = {}
        hookF_weights_wr = {}
        idx = 0
        if args.diffkurt:
            args.w_kurtosis_target = [1.8, 1.8, 1.8, 1.8,
                                      1.8, 1.8, 1.4, 1.8, 1.8,
                                      1.8, 1.4, 1.4, 1.4, 1.4,
                                      1.8, 1.2, 1.4, 1.2, 1.2]
        else:
            args.w_kurtosis_target = [args.w_kurtosis_target] * len(weight_to_hook)
        for name, w_tensor in weight_to_hook.items():
            hookF_weights[name] = KurtosisWeight(w_tensor, name, kurtosis_target=args.w_kurtosis_target[idx],
                                                 k_mode=args.kurtosis_mode)
            if args.w_l2_reg:
                hookF_weights_L2[name] = RidgeRegularization(w_tensor, name)
            if args.w_wr_reg:
                hookF_weights_wr[name] = WeightRegularization(w_tensor, name)
            idx = idx + 1
        ############ kurt ###########
        # compute output
        output_stud = model_stud(images)
        output_teacher = model_teacher(images)
        ############ KL div loss ###########
        if args.react:
            args.alpha = args.alpha
            args.beta = 0
            args.w_lambda_ce = 0
            loss_kl = 0
        else:
            loss_kl = criterion_kl(output_stud, output_teacher,model_stud,model_teacher,args.temperature) * args.beta
        loss_kl_c = criterion_kl_c(output_stud, output_teacher) * args.alpha
        ############ cross entrophy loss ###########
        orig_loss = criterion(output_stud, target) * args.w_lambda_ce
        ############ kurt loss ###########
        w_k_scale = 0
        w_kurtosis_regularization = 0
        if args.w_kurtosis and args.kurtepoch <= epoch:
            w_temp_values = []
            w_temp_kld_values = []
            w_kurtosis_loss = 0
            for w_kurt_inst in hookF_weights.values():
                w_kurt_inst.fn_regularization()
                w_temp_values.append(w_kurt_inst.kurtosis_loss)
                w_temp_kld_values.append(w_kurt_inst.KLDiv_loss)
            if args.kurtosis_mode == 'sum':
                w_kurtosis_loss = reduce((lambda a, b: a + b), w_temp_values)
            elif args.kurtosis_mode == 'avg':
                w_kurtosis_loss = reduce((lambda a, b: a + b), w_temp_values)
                w_kurtosis_loss = w_kurtosis_loss / len(weight_to_hook)
            elif args.kurtosis_mode == 'max':
                w_kurtosis_loss = reduce((lambda a, b: max(a, b)), w_temp_values)
            w_kurtosis_regularization = (10 ** w_k_scale) * args.w_lambda_kurtosis * w_kurtosis_loss
        ############ kurt loss ###########

        loss = (loss_kl) +loss_kl_c + orig_loss + ( w_kurtosis_regularization)

        acc1, acc5 = utils.accuracy(output_stud, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        losses_ce.update(orig_loss.item(), images.size(0))
        if args.w_kurtosis:
            losses_kurt.update(w_kurtosis_regularization.item(), images.size(0))
        if not args.react:
            losses_kl.update(loss_kl.item(), images.size(0))
        losses_kl_c.update(loss_kl_c.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            remain_epoch = args.epochs - epoch
            remain_iters = remain_epoch * len(train_loader) + (len(train_loader) - i)
            remain_seconds = remain_iters * batch_time.get_avg()
            seconds = (remain_seconds//1) % 60
            minutes = (remain_seconds//(1*60)) % 60
            hours = (remain_seconds//(1*60*60)) % 24
            days = (remain_seconds//(1*60*60*24))
            time_stamp = ""
            if (days > 0): time_stamp += "{} days, ".format(days)
            if (hours > 0) : time_stamp += "{} hr, ".format(hours)
            if (minutes > 0) : time_stamp += "{} min, ".format(minutes)
            if (seconds > 0) : time_stamp += "{} sec, ".format(seconds)
            msglogger.info(">>>>>>>>>>>> Remaining Times: {}  <<<<<<<<<<<<<<<<<".format(time_stamp) )


    writer.add_scalar("Train Loss", loss.item(), epoch)
    writer.add_scalar("Train Acc1", top1.avg, epoch)
    writer.add_scalar("Train Acc5", top5.avg, epoch)

def validate(val_loader, model, criterion, args, epoch):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    progress = utils.ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        msglogger, prefix='Test: ')

    model.eval().cuda()
    
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        writer.add_scalar("Val Loss", loss.item(), epoch)
        writer.add_scalar("Val Acc1", top1.avg, epoch)
        writer.add_scalar("Val Acc5", top5.avg, epoch)

    return top1.avg

if __name__ == '__main__':
    main()
