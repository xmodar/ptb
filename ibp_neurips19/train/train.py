"""Helper neural network training module."""

from pathlib import Path
from time import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from .utils import AverageMeter, Flatten, compute_accuracy, manual_seed

__all__ = ['train_classifier', 'one_epoch', 'small_cnn']


def small_cnn(pretrained=False):
    """Define a small CNN."""
    if pretrained:
        raise NotImplementedError('We don\'t have pretrained weights.')
    activation = nn.ReLU(inplace=True)
    net = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2),
        activation,
        nn.Conv2d(16, 32, 4, stride=1),
        activation,
        Flatten(),
        nn.Linear(3200, 100),
        activation,
        nn.Linear(100, 10),
    )
    return net


models.__dict__['small_cnn'] = small_cnn


def train_classifier(evaluate_only, dataset, model, pretrained, epochs,
                     batch_size, learning_rate, momentum, weight_decay, gpu,
                     jobs, checkpoint, resume, seed):
    """Train and/or evaluate a network."""
    manual_seed(seed, benchmark_otherwise=True)
    resume = Path(resume if resume else '')
    checkpoint = Path(checkpoint if checkpoint else '')

    # create model
    if pretrained:
        print(f'=> using pre-trained model {model}')
        net = models.__dict__[model](pretrained=True)
    else:
        print(f'=> creating model {model}')
        net = models.__dict__[model]()

    using_cuda = False
    if torch.cuda.is_available():
        using_cuda = True
        if torch.cuda.device_count() == 1:
            gpu = 0
        if gpu is not None:
            print(f'Use GPU: {gpu} for training')
            torch.cuda.set_device(gpu)
            net = net.cuda(gpu)
        else:
            # DataParallel will divide and allocate batch_size to all GPUs
            print(f'Use {torch.cuda.device_count()} GPUs for training')
            if model.startswith('alexnet') or model.startswith('vgg'):
                net.features = torch.nn.DataParallel(net.features)
                net.cuda()
            else:
                net = torch.nn.DataParallel(net).cuda()

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # TODO: move to device

    optimizer = torch.optim.SGD(
        net.parameters(),
        learning_rate,
        momentum=momentum,
        weight_decay=weight_decay)

    # optionally resume from a checkpoint
    best_acc1 = 0
    start_epoch = 0
    if resume.is_file():
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(
            resume, checkpoint['epoch']))
    elif resume != Path():
        print("=> no checkpoint found at '{}'".format(resume))

    # Data loading code
    datasets_dir = Path.home() / '.torch/datasets'
    means = {
        'MNIST': [0.1307],
        'SVHN': [0.5071, 0.4867, 0.4408],
        'CIFAR10': [0.4915, 0.4823, 0.4468],
        'CIFAR100': [0.5072, 0.4867, 0.4412],
    }
    stds = {
        'MNIST': [0.3081],
        'SVHN': [0.2675, 0.2565, 0.2761],
        'CIFAR10': [0.2470, 0.2435, 0.2616],
        'CIFAR100': [0.2673, 0.2564, 0.2762],
    }
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means[dataset], stds[dataset]),
    ])
    train_dataset = datasets.__dict__[dataset](
        datasets_dir, train=True, transform=transform, download=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=jobs if using_cuda else 0,
        pin_memory=using_cuda,
        drop_last=True,
    )

    val_dataset = datasets.__dict__[dataset](
        datasets_dir, train=False, transform=transform, download=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=jobs if using_cuda else 0,
        pin_memory=using_cuda,
        drop_last=False,
    )

    process = lambda loader, opt: one_epoch(
        loader, net, criterion, opt, using_cuda, gpu,)
    progress = process(val_loader, None)
    val_loss = progress['Loss'].avg
    val_acc = progress['Acc@1'].avg
    print(f'Test[{val_loss:.4e}: {val_acc:6.2f}%]')
    if evaluate_only:
        return

    for epoch in range(start_epoch, epochs):
        # decay the learning rate by 10 every 30 epochs
        if epoch % 30 == 0:
            lr = learning_rate * (0.1**(epoch // 30))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # train for one epoch and evaluate on validation set
        progress = process(train_loader, optimizer)
        train_loss = progress['Loss'].avg
        train_acc = progress['Acc@1'].avg

        progress = process(val_loader, None)
        val_loss = progress['Loss'].avg
        val_acc = progress['Acc@1'].avg

        print(f'[{epoch}@{lr:.4e}] '
              f'Train[{train_loss:.4e}: {train_acc:6.2f}%] '
              f'Test[{val_loss:.4e}: {val_acc:6.2f}%]')

        # remember best acc@1 and save checkpoint
        if val_acc >= best_acc1:
            best_acc1 = val_acc
            if checkpoint != Path():
                torch.save({
                    'epoch': epoch + 1,
                    'net': net,
                    'state_dict': net.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, checkpoint)


def one_epoch(train_loader, net, criterion, optimizer, using_cuda, gpu):
    """Perform one training epoch."""
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to train mode
    is_training = optimizer is not None
    net.train(is_training)

    with torch.set_grad_enabled(is_training):
        end = time()
        for inputs, target in train_loader:
            # measure data loading time
            data_time.update(time() - end)

            if gpu is not None:
                inputs = inputs.cuda(gpu, non_blocking=True)
            if using_cuda:
                target = target.cuda(gpu, non_blocking=True)

            # compute output
            output = net(inputs)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = compute_accuracy(  # pylint: disable=E0632
                output, target, top_k=(1, 5))
            losses.update(float(loss), inputs.size(0))
            top1.update(float(acc1), inputs.size(0))
            top5.update(float(acc5), inputs.size(0))

            # compute gradient and do SGD step
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  # TODO: pass training closure

            # measure elapsed time
            batch_time.update(time() - end)
            end = time()

    return {x.name: x for x in (batch_time, data_time, losses, top1, top5)}
