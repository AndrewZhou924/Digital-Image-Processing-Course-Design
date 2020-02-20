import pickle
import os
from tqdm import tqdm
import argparse

# use cuda or not
def str2bool(s):
    return s in ['1', 'True', 'TRUE']
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', required=True, type=str2bool, help='Use cuda or not')
args = parser.parse_args()
if args.cuda:
    import cupy as np
else:
    import numpy as np
# ugly code here

def unpickle(file: str) -> dict:
    '''
    :param file: path to one cifar10 item
    :return: dict of one cifar10 item
    '''
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d

def cifar10(root: str) -> (np.array, np.array, np.array, np.array):
    '''
    :param root: root path for cifar 10 directory
    :return: tuple of (train_datas, train_labels, test_datas, test_labels)
    '''
    train_paths = [os.path.join(root, i) for i in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']]
    test_path = os.path.join(root, 'test_batch')

    train_datas_list, train_labels_list = [], []

    cuda = np.__name__=='cupy'

    for path in train_paths:
        d = unpickle(path)

        if cuda:
            train_datas_list.append(np.array((d[b'data']/255).tolist(), dtype=np.float32)) # tolist() may cost a lot of time, so don't use it when we not use cuda
        else:
            train_datas_list.append(np.array(d[b'data']/255, dtype=np.float32))

        train_labels_list += d[b'labels']
    train_labels = np.array(train_labels_list, dtype=np.int32)

    train_datas = np.concatenate(train_datas_list)

    d = unpickle(test_path)

    if cuda:
        test_datas = np.array((d[b'data']/255).tolist(), dtype=np.float32) # tolist()...
    else:
        test_datas = np.array(d[b'data']/255, dtype=np.float32)

    test_labels = np.array(d[b'labels'], dtype=np.int32)
    return (train_datas, train_labels, test_datas, test_labels)

def data_check(datas: np.array, labels: np.array) -> None:
    '''
    check datas with labels
    :param datas: train or test data in numpy array
    :param labels: train or test label in numpy array
    :return: None
    '''
    import random
    import matplotlib.pyplot as plt

    label_class_map = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    data_nums = len(labels)
    datas = np.reshape(datas, (-1, 3, 32, 32))
    while 1:
        index = random.randint(0, data_nums)
        data = np.transpose(datas[index], (1, 2, 0))
        label = labels[index]
        plt.imshow(data)
        plt.title(label_class_map[label])
        plt.show()

class NearestNeighbor:
    def train(self, x: np.array, y: np.array) -> None:
        '''
        train process for KNN
        :param x: train data x
        :param y: train label y
        :return: None
        '''
        self.x = x
        self.y = y

    def forward(self, x: np.array, method: str) -> None:
        '''
        feature extract and argsort
        :param x: test data x
        :param method: method for feature extract
        :return: None
        '''

        assert method in ['l1', 'l2']

        distance = []

        if method=='l1':
            # for L1 distance
            data_num = x.shape[0]
            for i in tqdm(range(data_num)):
                distance.append(np.sum(np.abs(self.x - x[i]), axis = 1))
            distance = np.array(distance)

        if method=='l2':
            # for L2 distance
            distance = -2 * x.dot(self.x.T) + np.sum(np.square(x), axis=1, keepdims=1) + np.sum(np.square(self.x), axis=1).T

        self.sorted = []
        for i in tqdm(range(distance.shape[0])): # for not cuda oom, use forloop instead of np.argsort the whole array
            self.sorted.append(np.argsort(distance[i]).tolist())

    def predict(self, k: int) -> np.array:
        '''
        get result for knn
        :param k: knn's k
        :return: y_pred
        '''

        y_pred = np.zeros(len(self.sorted), dtype=self.y.dtype)
        for i in range(len(self.sorted)):
            counter = dict()
            for j in range(k):
                classes = self.y[self.sorted[i][j]].item()
                counter[classes] = counter.get(classes, 0) + 1
            sorted_counter = sorted(counter.items(), key=lambda x:x[1], reverse=True)
            y_pred[i] = sorted_counter[0][0]
        return y_pred

    def criterion(self, y_gt: np.array, y_pred: np.array) -> float:
        '''
        critertion for KNN Model, return the accuracy for predict result
        :param y_gt: ground truth in numpy array
        :param y_pred: predict result in numpy array
        :return: accuracy in float
        '''
        assert y_gt.shape == y_pred.shape

        total = y_gt.shape[0]

        correct = np.equal(y_gt, y_pred).sum()

        return correct/total

if __name__=='__main__':
    train_datas, train_labels, test_datas, test_labels = cifar10('data/cifar-10-batches-py/')
    # data_check(train_datas, train_labels) # check if datas and labels match
    knn_model = NearestNeighbor()
    knn_model.train(train_datas, train_labels)

    knn_model.forward(test_datas, method='l1')
    test_predictions = knn_model.predict(k=10)
    acc = knn_model.criterion(test_labels, test_predictions)
    print('final acc: {}'.format(acc))

