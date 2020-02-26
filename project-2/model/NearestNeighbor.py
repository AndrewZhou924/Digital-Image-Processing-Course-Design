# import numpy as np
import cupy as np
from tqdm import tqdm
from collections import Counter
from skimage.feature import hog
import matplotlib.pyplot as plt
import importlib

class NearestNeighbor:
    def __init__(self, cuda:bool) -> None:
        self.cuda = cuda
        
        # if cuda:
            # import cupy as np
            # np = importlib.reload(cupy)
        # else:
            # import numpy as np
            # np = importlib.reload(numpy)

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