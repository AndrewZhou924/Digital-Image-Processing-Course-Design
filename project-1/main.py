import pickle
import os
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import time
from collections import Counter
from skimage.feature import hog
import numpy

# use cuda or not
def str2bool(s):
    return s in ['1', 'True', 'TRUE']
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', required=True, type=str2bool, help='Use cuda or not')
parser.add_argument('--dir', required=True, type=str, help='Dataset directory')
parser.add_argument('--use_hog', required=False, default=False, type=str2bool, help='Use hog or not')
parser.add_argument('--method', required=False, default='l1', type=str, help='method for feature extract')
parser.add_argument('--maxk', required=False, default=20, type=int, help='max k')
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

def cifar10(root: str, use_hog=True) -> (np.array, np.array, np.array, np.array):
    '''
    :param root: root path for cifar 10 directory
    :param hog: use hog or not
    :return: tuple of (train_datas, train_labels, test_datas, test_labels)
    '''
    train_paths = [os.path.join(root, i) for i in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']]
    test_path = os.path.join(root, 'test_batch')

    train_datas_list, train_labels_list = [], []

    cuda = np.__name__=='cupy'

    if use_hog:
        train_features = []
        for path in train_paths:
            d = unpickle(path)
            train_datas_list.append(d[b'data'])
            train_labels_list += d[b'labels']

        train_data = numpy.concatenate(train_datas_list).reshape((-1, 3, 32, 32))
        train_data = numpy.transpose(train_data, (0, 2, 3, 1))

        for i in tqdm(range(train_data.shape[0])):
            train_features.append(hog(train_data[i], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), multichannel=True))
        train_features = numpy.array(train_features, dtype=numpy.float32)
        train_labels = numpy.array(train_labels_list, dtype=np.int32)

        d = unpickle(test_path)
        test_datas = numpy.reshape(d[b'data'], (-1, 3, 32, 32))
        test_datas = numpy.transpose(test_datas, (0, 2, 3, 1))
        test_features = []
        for i in tqdm(range(test_datas.shape[0])):
            test_features.append(hog(test_datas[i], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), multichannel=True))
        test_features = numpy.array(test_features, dtype=numpy.float32)
        test_labels = numpy.array(d[b'labels'], dtype=numpy.int32)

        if cuda:
            train_features = np.array(train_features.tolist(), dtype=np.float32)
            train_labels = np.array(train_labels.tolist(), dtype=np.int32)
            test_features = np.array(test_features.tolist(), dtype=np.float32)
            test_labels = np.array(test_labels.tolist(), dtype=np.int32)

        # print(train_features.shape, train_labels.shape, test_features.shape, test_labels.shape)
        return train_features, train_labels, test_features, test_labels

    for path in train_paths:
        d = unpickle(path)

        if cuda:
            train_datas_list.append(np.array((d[b'data']).tolist(), dtype=np.int32)) # tolist() may cost a lot of time, so don't use it when we not use cuda
        else:
            train_datas_list.append(np.array(d[b'data'], dtype=np.int32))

        train_labels_list += d[b'labels']
    train_labels = np.array(train_labels_list, dtype=np.int32)

    train_datas = np.concatenate(train_datas_list)

    d = unpickle(test_path)

    if cuda:
        test_datas = np.array((d[b'data']).tolist(), dtype=np.int32) # tolist()...
    else:
        test_datas = np.array(d[b'data'], dtype=np.int32)

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
        self.x = x[:10000, :]
        self.y = y[:10000]

    def forward(self, x: np.array, method: str) -> None:
        '''
        feature extract and argsort
        :param x: test data x
        :param method: method for feature extract
        :return: None
        '''

        assert method in ['l1', 'l2', 'cosine']

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

        if method=='cosine':
            # for cosine distance
            distance = 1 - x.dot(self.x.T) / np.linalg.norm(x, axis=1, keepdims=1) / np.linalg.norm(self.x, axis=1).T

        # self.distance = distance
        self.sorted = []
        for i in tqdm(range(distance.shape[0])): # for not cuda oom, use forloop instead of np.argsort the whole array
            self.sorted.append(np.argsort(distance[i]).tolist())

    def predict(self, k: int) -> np.array:
        '''
        get result for knn
        :param k: knn's k
        :return: y_pred
        '''
        # y_pred = np.zeros(self.distance.shape[0], dtype=self.y.dtype)
        # for i in range(self.distance.shape[0]):
        #     curr_distance = self.distance[i]
        #     min_idx = np.argpartition(curr_distance, k)[0:k]
        #     votes = self.y[min_idx]
        #     labels_count = np.bincount(votes)
        #     y_pred[i] = np.argmax(labels_count)
        # return y_pred
        # another way to get predict result. np.argpartition and np.bincount

        y_pred = np.zeros(len(self.sorted), dtype=self.y.dtype)
        for i in range(len(self.sorted)):
            classes = self.y[self.sorted[i][:k]].tolist()
            counter = Counter(classes)
            y_pred[i] = counter.most_common()[0][0]
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

train_datas, train_labels, test_datas, test_labels = cifar10(args.dir, args.use_hog)
# data_check(train_datas, train_labels) # check if datas and labels match
knn_model = NearestNeighbor()
knn_model.train(train_datas, train_labels)

# caculate forward time
start = time.time()
knn_model.forward(test_datas, method=args.method)
end = time.time()
print('knn model forward time cost: {}'.format(end-start))

# search K
acc = list()
for k in tqdm(range(1, args.maxk+1)):
    test_predictions = knn_model.predict(k)
    acc.append(knn_model.criterion(test_labels, test_predictions))
start = time.time()
print('knn model traverse k time cost: {}'.format(start-end))

# draw image
x = list(range(1, args.maxk+1))
plt.plot(x, acc)
plt.title('method={} maxk={} use_hog={}'.format(args.method, args.maxk, args.use_hog))
plt.savefig('{}-{}-{}.jpg'.format(args.method, args.maxk, 'HOG' if args.use_hog else 'NO_HOG'))

print('max accuracy: {}'.format(max(acc)))
