from jittor.dataset.dataset import Dataset
import jittor.transform as transform
import pickle
import os
import numpy as np
from PIL import Image

def unpickle(file: str) -> dict:
    '''
    :param file: path to one cifar10 item
    :return: dict of one cifar10 item
    '''
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d

class CIFAR10(Dataset):
    def __init__(self, data_root='./data/cifar-10-batches-py/', train=True, batch_size=32, shuffle=False, transform=None):
        super().__init__()

        if train:
            path = [os.path.join(data_root, i) for i in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']]
        else:
            path = [os.path.join(data_root, i) for i in ['test_batch']]

        self.transform = None


        self.data, self.label = [], []
        for p in path:
            d = unpickle(p)
            self.data.append(d[b'data'])
            self.label.append(d[b'labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.label = np.concatenate(self.label, axis=0)

        self.total_len = self.__len__()
        self.set_attrs(batch_size = batch_size, total_len = self.total_len, shuffle = shuffle, transform=transform)


    def __getitem__(self, idx):
        data, label = self.data[idx], self.label[idx]
        data = Image.fromarray(data)

        if self.transform is not None:
            data = self.transform(data)

        return data, label

    def __len__(self):
        return self.data.shape[0]

if __name__=='__main__':
    dataloader = CIFAR10(train=False, batch_size=2)
    for i, (inputs, targets) in enumerate(dataloader):
        print(inputs.shape, targets.shape)
        print(inputs.dtype, targets.dtype)
        break