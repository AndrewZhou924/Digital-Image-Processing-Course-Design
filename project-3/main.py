import nn
import optim
from dataloader import Cifar10, Test
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt

np.random.seed(0)
random.seed(0)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [
            nn.Dense(3072, 100), # layer1
            nn.BatchNorm(100),
            nn.ReLU(),
            nn.Dropout(0.75),
            nn.Dense(100, 100), # layer2
            nn.BatchNorm(100),
            nn.ReLU(),
            nn.Dropout(0.75),
            nn.Dense(100, 100), # layer3
            nn.BatchNorm(100),
            nn.ReLU(),
            nn.Dropout(0.75),
            nn.Dense(100, 100), # layer4
            nn.BatchNorm(100),
            nn.ReLU(),
            nn.Dropout(0.75),
            nn.Dense(100, 10), # layer5
        ]
    def initialize(self):
        for layer in self.layers:
            if layer.__repr__()=='Dense':
                layer.parameters[0] = nn.init.xavier_normal(layer.parameters[0])
                layer.parameters[1] = nn.init.constant(layer.parameters[1], 0.1)

def train(model, trainloader, criterion, optimizer):

    model.train()
    loss = 0
    correct, total = 0, 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader), total=len(trainloader)):
        outputs = model(inputs)
        preds = np.argmax(outputs, axis=1)
        loss += criterion(outputs, targets)
        model.backward(criterion.backward())
        model.update(optimizer)

        correct += np.sum(preds == targets)
        total += inputs.shape[0]
    acc = correct / total
    loss = loss / len(trainloader)
    print('train loss : {} | train acc : {}'.format(loss, acc))
    return loss, acc

def test(model, testloader, criterion):

    model.eval()
    loss = 0
    correct, total = 0, 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(testloader), total=len(testloader)):
        outputs = model(inputs)
        loss += criterion(outputs, targets)
        preds = np.argmax(outputs, axis=1)
        correct += np.sum(preds == targets)
        total += inputs.shape[0]
    acc = correct / total
    loss = loss / len(testloader)
    print('test loss: {} | test acc: {}'.format(loss, acc))
    return loss, acc

if __name__=='__main__':
    trainloader = Cifar10('./data/cifar-10-batches-py/', batch_size=32, phase='train', shuffle='True')
    testloader = Cifar10('./data/cifar-10-batches-py/', batch_size=100, phase='test', shuffle='False')

    model = Model()
    model.initialize()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5-4)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoch_list, train_loss_list, train_acc_list, test_loss_list, test_acc_list = [], [], [], [], []
    early_stop = 3
    for i in range(100):
        train_loss, train_acc = train(model, trainloader, criterion, optimizer)
        test_loss, test_acc = test(model, testloader, criterion)

        epoch_list.append(i)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        if len(test_acc_list)>1 and test_acc<test_acc_list[-2]:
            early_stop -= 1
        if early_stop == 0:
            plt.figure()
            plt.plot(epoch_list, train_loss_list, label='train_loss', marker='o')
            for a,b in zip(epoch_list, train_loss_list):
                plt.text(a, b, "%.3f" % b)
            plt.plot(epoch_list, test_loss_list, label='test_loss', marker='o')
            for a,b in zip(epoch_list, test_loss_list):
                plt.text(a, b, "%.3f" % b)
            plt.legend()
            plt.savefig('loss.jpg')

            plt.figure()
            plt.plot(epoch_list, train_acc_list, label='train_acc', marker='o')
            for a,b in zip(epoch_list, train_acc_list):
                plt.text(a, b, "%.3f" % b)
            plt.plot(epoch_list, test_acc_list, label='test_acc', marker='o')
            for a,b in zip(epoch_list, test_acc_list):
                plt.text(a, b, "%.3f" % b)
            plt.legend()
            plt.savefig('acc.jpg')
            break