import jittor as jt
import jittor.nn as nn
import jittor.transform as transform
import numpy as np
from dataloader import CIFAR10
import argparse
from models import *
from tqdm import tqdm

def train(model, trainloader, optimizer):
    model.train()
    loss_ = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        outputs = model(inputs)
        predicts = np.argmax(outputs.data, axis=1)
        correct += np.sum(targets.data==predicts)
        total += targets.shape[0]
        loss = nn.cross_entropy_loss(outputs, targets)
        optimizer.step(loss)
        loss_ += loss.data[0]
    loss_ /= len(trainloader)
    accuracy = correct / total
    print("Train Loss: {} | Train Acc: {}".format(loss_, accuracy))

def test(model, testloader):
    model.eval()

    loss_ = 0
    total, correct = 0, 0

    for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
        outputs = model(inputs)
        predicts = np.argmax(outputs.data, axis=1)
        correct += np.sum(targets.data==predicts)
        total += targets.shape[0]
        loss = nn.cross_entropy_loss(outputs, targets)
        loss_ += loss.data[0]
    loss_ /= len(testloader)
    accuracy = correct / total
    print("Test Loss: {} | Test Acc: {}".format(loss_, accuracy))

if __name__=='__main__':
    jt.flags.use_cuda = 1

    train_transform = transform.Compose([
        transform.RandomHorizontalFlip(),
        transform.RandomCropAndResize(32),
        transform.ImageNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    test_transform = transform.Compose([
        transform.ImageNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    trainloader = CIFAR10(train=True, shuffle=True, batch_size=32, transform=train_transform)
    testloader = CIFAR10(train=False, shuffle=False, transform=test_transform)

    model = VGG16_bn()
    optimizer = nn.SGD(model.parameters(), 0.1, 0.9, 5e-4)

    for epoch in range(100):
        train(model, trainloader, optimizer)
        test(model, testloader)