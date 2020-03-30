import jittor as jt
import jittor.nn as nn
import jittor.transform as transform
import numpy as np
from dataloader import CIFAR10
import argparse
from models import VGG, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from tqdm import tqdm
import tensorboardX
import logging
import os
import sys
import math
import random

np.random.seed(1020)
random.seed(1020)

def train(epoch, model, trainloader, optimizer):
    model.train()
    loss_ = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader), total=math.ceil(len(trainloader)/trainloader.batch_size)):
        outputs = model(inputs)
        predicts = np.argmax(outputs.data, axis=1)
        correct += np.sum(targets.data==predicts)
        total += targets.shape[0]
        loss = nn.cross_entropy_loss(outputs, targets)
        optimizer.step(loss)
        loss_ += loss.data[0]
    loss_ /= len(trainloader)
    accuracy = correct / total
    print("Train Epoch: {} | Train Loss: {} | Train Acc: {}".format(epoch, loss_, accuracy))
    return loss_, accuracy

def test(epoch, model, testloader):
    model.eval()

    loss_ = 0
    total, correct = 0, 0

    for batch_idx, (inputs, targets) in tqdm(enumerate(testloader), total=math.ceil(len(testloader)/testloader.batch_size)):
        outputs = model(inputs)
        predicts = np.argmax(outputs.data, axis=1)
        correct += np.sum(targets.data==predicts)
        total += targets.shape[0]
        loss = nn.cross_entropy_loss(outputs, targets)
        loss_ += loss.data[0]
    loss_ /= len(testloader)
    accuracy = correct / total
    print("Test Epoch : {} | Test Loss: {} | Test Acc: {}".format(epoch, loss_, accuracy))
    return loss_, accuracy

def get_model(name: str):
    name = name.lower()
    if name=='vgg11':
        return VGG('VGG11')
    elif name=='resnet18':
        return ResNet18()
    elif name=='resnet34':
        return ResNet34()
    elif name=='resnet50':
        return ResNet50()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required=True, type=str, help='log dir')
    parser.add_argument('--epoch_num', default=100, type=int, help='epoch nums')
    parser.add_argument('--learning_rate', required=True, type=float, help='learning_rate')
    parser.add_argument('--model', default='resnet18', type=str, help='neural network')
    parser.add_argument('--cuda', action='store_true',help='to use cuda')

    args = parser.parse_args()

    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)

    logging.basicConfig(filename=os.path.join(args.logdir, 'log.txt'), level=logging.INFO)
    logging.info(sys.argv)

    jt.flags.use_cuda = 1 if args.cuda else 0

    train_transform = transform.Compose([
        transform.RandomHorizontalFlip(),
        transform.RandomCropAndResize(32, scale=(0.5, 1)),
        transform.ImageNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    test_transform = transform.Compose([
        transform.ImageNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    trainloader = CIFAR10(train=True, shuffle=True, batch_size=64, transform=train_transform)
    testloader = CIFAR10(train=False, shuffle=False, batch_size=100, transform=test_transform)

    model = get_model(args.model)

    optimizer = nn.SGD(parameters=model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)

    summary_writer = tensorboardX.SummaryWriter(logdir=args.logdir)

    decay_lr_at = [int(args.epoch_num * i) for i in [0.25, 0.5, 0.75]]

    max_acc = 0.
    for epoch in range(args.epoch_num):
        if epoch in decay_lr_at:
            optimizer.lr *= 0.1
        train_loss, train_acc = train(epoch, model, trainloader, optimizer)
        test_loss, test_acc = test(epoch, model, testloader)

        summary_writer.add_scalar('Train Loss', train_loss, epoch)
        summary_writer.add_scalar('Train Acc', train_acc, epoch)
        summary_writer.add_scalar('Test Loss', test_loss, epoch)
        summary_writer.add_scalar('Test Acc', test_acc, epoch)

        if test_acc > max_acc:
            max_acc = test_acc
            model.save(os.path.join(
                args.logdir,
                '{}-{}-{}.pkl'.format(args.model, epoch, max_acc)
            ))