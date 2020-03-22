import random
import pickle
import os
import math
import numpy as np

class Cifar10:
    def __init__(self, root, batch_size, phase='train', shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle

        if phase=='train':
            path_list = [os.path.join(root, i) for i in
                           ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']]
        else:
            path_list = [os.path.join(root, 'test_batch')]
        data_list, label_list = [], []
        for path in path_list:
            d = self.unpickle(path)
            data_list.append(np.array(d[b'data']/255, dtype=np.float32))
            label_list+=d[b'labels']

        self.data = np.concatenate(data_list)
        self.label = np.array(label_list, dtype=np.int32)
        self.length = self.data.shape[0]

    def __iter__(self):
        index = list(range(self.length))
        if self.shuffle:
            random.shuffle(index)
        data_batch_list = []
        label_batch_list = []
        counter = 0
        for idx in index:
            data_batch_list.append(self.data[idx])
            label_batch_list.append(self.label[idx])
            counter += 1
            if counter==self.batch_size or idx==index[-1]:
                data_batch = np.array(data_batch_list, dtype=np.float32)
                label_batch = np.array(label_batch_list, dtype=np.int32)
                data_batch_list, label_batch_list = [], []
                counter = 0
                yield (data_batch, label_batch)

    def __len__(self):
        return math.ceil(self.length / self.batch_size)

    def unpickle(self, file: str) -> dict:
        with open(file, 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')
        return d

class Test:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.num = 1000000
        self.x = np.random.randn(self.num, 2)
        self.y = np.array(self.x[:, 1]>self.x[:, 0], dtype=np.int32)

    def __iter__(self):
        counter = 0
        data_batch_list = []
        label_batch_list = []
        for i in range(self.num):
            data_batch_list.append(self.x[i])
            label_batch_list.append(self.y[i])
            counter += 1
            if counter == self.batch_size or i==self.num-1:
                data_batch = np.array(data_batch_list, dtype=np.float32)
                label_batch = np.array(label_batch_list, dtype=np.int32)
                data_batch_list, label_batch_list = [], []
                counter = 0
                yield (data_batch, label_batch)

    def __len__(self):
        return math.ceil(self.num / self.batch_size)

    def len(self):
        return self.__len__()

if __name__=='__main__':
    import matplotlib.pyplot as plt

    label_class_map = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    dataset = Cifar10(root='./data/cifar-10-batches-py', batch_size=2, phase='train', shuffle=False)
    for batch_idx, (inputs, targets) in enumerate(dataset):
        print(batch_idx, inputs, targets)
        image = np.array(inputs[0]*255, dtype=np.uint8)
        image = np.resize(image, (3, 32, 32))
        image = np.transpose(image, (1, 2, 0))
        plt.title(label_class_map[targets[0]])
        plt.imshow(image)
        plt.show()