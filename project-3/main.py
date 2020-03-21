import nn
import optim
from dataloader import Cifar10, Test
from tqdm import tqdm
import numpy as np

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [
            nn.Dense(2, 10),
            nn.BatchNorm(10),
            nn.ReLU(),
            nn.Dense(10, 2),
            nn.BatchNorm(2),
        ]

def train(model, trainloader, criterion, optimizer):

    model.train()
    loss = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        outputs = model(inputs)
        loss += criterion(outputs, targets)
        model.backward(criterion.backward())
        model.update(optimizer)
    print('final loss : {}'.format(loss / len(trainloader)))

def test(model, testloader):

    model.eval()
    correct, total = 0, 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
        outputs = model(inputs)
        preds = np.argmax(outputs, axis=1)
        correct += np.sum(preds == targets)
        total += inputs.shape[0]
    print('final acc: {}'.format(correct / total))

if __name__=='__main__':
    trainloader = Test(10, np)
    testloader = Test(10, np)

    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)

    for i in range(100):
        train(model, trainloader, criterion, optimizer)
        test(model, testloader)