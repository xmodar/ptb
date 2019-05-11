"""Helper neural network training module."""

from collections import OrderedDict
from pathlib import Path
from time import time

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from ..datasets import get_loader
from ..models import get_model
from .utils import (AverageMeter, compute_accuracy, get_device_order,
                    manual_seed)

__all__ = ['train_classifier', 'one_epoch']


def train_classifier(evaluate_only, dataset, model, pretrained, learning_rate,
                     momentum, weight_decay, epochs, batch_size, jobs,
                     checkpoint, resume, log_dir, seed):
    """Train and/or evaluate a network."""
    manual_seed(seed, benchmark_otherwise=True)
    resume = Path(resume if resume else '')
    checkpoint = Path(checkpoint if checkpoint else '')
    get_lr = lambda epoch: learning_rate * (0.1**(epoch // 30))

    # get available cuda devices ordered by total memory capacity
    devices = get_device_order()
    if devices:
        print(f'=> using {len(devices)} GPU(s)')
        device = torch.device(f'cuda:{devices[0]}')
    else:
        device = torch.device('cpu')

    def to_device(*tensors, non_blocking=True):
        return [t.to(device, non_blocking=non_blocking) for t in tensors]

    # Data loading code
    cuda = len(devices) > 0
    train_loader = get_loader(dataset, True, batch_size, cuda, jobs)
    val_loader = get_loader(dataset, False, batch_size, cuda, jobs)

    # create the model
    if pretrained:
        print(f'=> using pre-trained model {model}')
    else:
        print(f'=> creating model {model}')
    net = get_model(model, pretrained)
    keys = net.state_dict(keep_vars=True).keys()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    to_device(net, criterion, non_blocking=False)
    optimizer = torch.optim.SGD(
        net.parameters(),
        learning_rate,
        momentum=momentum,
        weight_decay=weight_decay)

    # define a colsure wrapping one_epoch()
    def process(loader, optimizer=None):
        return one_epoch(loader, net, criterion, optimizer, to_device)

    # optionally resume from a checkpoint
    best_acc1 = 0
    start_epoch = 0
    if resume.is_file():
        print("=> loading checkpoint '{}'".format(resume))
        state = torch.load(resume)
        start_epoch = state['epoch']
        best_acc1 = state['best_acc1']
        net.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        print(f"=> loaded checkpoint '{resume}' (epoch {state['epoch']})")
    elif resume != Path():
        print(f"=> no checkpoint found at '{resume}'")

    # DataParallel will divide and allocate batch_size to all GPUs
    if len(devices) > 1:
        if model.startswith('alexnet') or model.startswith('vgg'):
            net.features = nn.DataParallel(net.features, devices, device)
        else:
            net = nn.DataParallel(net, devices, device)

    # evaluate the model before training
    progress = process(val_loader)
    val_loss = progress['Loss']
    val_acc = progress['Acc@1']
    print(f'Test[{val_loss}: {val_acc}%]')
    if evaluate_only:
        return

    if log_dir:
        writer = SummaryWriter(log_dir)
    lr = get_lr(start_epoch)
    for epoch in range(start_epoch, epochs):
        # decay the learning rate by 10 every 30 epochs
        if epoch % 30 == 0:
            lr = get_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # train for one epoch and evaluate on validation set
        train_progress = process(train_loader, optimizer)
        train_loss = train_progress['Loss']
        train_acc = train_progress['Acc@1']

        val_progress = process(val_loader)
        val_loss = val_progress['Loss']
        val_acc = val_progress['Acc@1']

        print(f'[{epoch + 1}@{lr:.4e}] '
              f'Train[{train_loss}: {train_acc}%] '
              f'Test[{val_loss}: {val_acc}%]')

        if log_dir:
            writer.add_scalar('Train/LearingRate', lr, epoch)
            for meter in train_progress.values():
                writer.add_scalar(f'Train/{meter.name}', meter.avg, epoch)
            for meter in val_progress.values():
                writer.add_scalar(f'Test/{meter.name}', meter.avg, epoch)

        # remember best acc@1 and save checkpoint
        if val_acc.avg >= best_acc1:
            best_acc1 = val_acc.avg
            if checkpoint != Path():
                parameters = net.state_dict().values()
                torch.save({
                    'epoch': epoch + 1,
                    'net': net,
                    'state_dict': OrderedDict(zip(keys, parameters)),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, checkpoint)
    if log_dir:
        writer.close()


def one_epoch(train_loader, net, criterion, optimizer, preporcess):
    """Perform one training epoch."""
    batch_time = AverageMeter('Time/BatchTotal', ':6.3f')
    data_time = AverageMeter('Time/BatchData', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to train mode
    is_training = optimizer is not None
    net.train(is_training)

    def compute_loss(inputs, targets, update_metrics):
        # compute output
        output = net(inputs)
        loss = criterion(output, targets)

        # measure accuracy and record loss
        if update_metrics:
            n = inputs.size(0)
            acc1, acc5 = compute_accuracy(  # pylint: disable=E0632
                output, targets, top_k=(1, 5))
            losses.update(float(loss), n)
            top1.update(float(acc1), n)
            top5.update(float(acc5), n)

        # compute gradient
        if is_training:
            optimizer.zero_grad()
            loss.backward()

        return loss

    with torch.set_grad_enabled(is_training):
        end = time()
        for inputs, targets in train_loader:
            # measure data loading time
            data_time.update(time() - end)

            # move data to device
            inputs, targets = preporcess(inputs, targets)

            first_time = True

            def closure():
                nonlocal first_time
                loss = compute_loss(
                    inputs,  # pylint: disable=W0640
                    targets,  # pylint: disable=W0640
                    first_time,
                )
                first_time = False
                return loss

            if is_training:
                optimizer.step(closure)
            else:
                closure()

            # measure elapsed time
            batch_time.update(time() - end)
            end = time()

    return {x.name: x for x in (batch_time, data_time, losses, top1, top5)}
