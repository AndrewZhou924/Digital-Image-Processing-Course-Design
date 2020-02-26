import pickle
import os
import numpy
import argparse
import matplotlib.pyplot as plt
import time
from collections import Counter
from tqdm import tqdm
from skimage.feature import hog
from model.NearestNeighbor import NearestNeighbor
from model.SVM import SVM
from model.softmax import softmax

# use cuda or not
def str2bool(s):
    return s in ['1', 'True', 'TRUE']

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', required=True, type=str2bool, help='Use cuda or not')
parser.add_argument('--dir', required=True, type=str, help='Dataset directory')
parser.add_argument('--use_hog', required=False, default=False, type=str2bool, help='Use hog or not')
parser.add_argument('--method', required=False, default='l1', type=str, help='method for feature extract')
parser.add_argument('--maxk', required=False, default=20, type=int, help='max k')
parser.add_argument('--model', required=False, default='knn', type=str, help='avaliable model: knn, svm, softmax')
parser.add_argument('--iters', required=False, default=200, type=int, help='max number of iterations')
parser.add_argument('--batch_size', required=False, default=200, type=int, help='batch_size')
parser.add_argument('--data_check', required=False, default=False, type=bool, help='check data or not')

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
            # train_features.append(hog(train_data[i], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), multichannel=True))
            train_features.append(hog(train_data[i], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), multichannel=True))
        train_features = numpy.array(train_features, dtype=numpy.float32)
        train_labels = numpy.array(train_labels_list, dtype=np.int32)

        d = unpickle(test_path)
        test_datas = numpy.reshape(d[b'data'], (-1, 3, 32, 32))
        test_datas = numpy.transpose(test_datas, (0, 2, 3, 1))
        test_features = []
        for i in tqdm(range(test_datas.shape[0])):
            # test_features.append(hog(test_datas[i], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), multichannel=True))
            test_features.append(hog(test_datas[i], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), multichannel=True))
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



train_datas, train_labels, test_datas, test_labels = cifar10(args.dir, args.use_hog)
print("==> train_datas shape: ", np.array(train_datas).shape)
print("==> train_labels shape: ", np.array(train_labels).shape)
print("==> test_datas shape: ", np.array(test_datas).shape)
print("==> test_labels shape: ", np.array(test_labels).shape)

# check whether datas and labels match or not
if args.data_check:
    data_check(train_datas, train_labels) 

# training model
assert args.model in ["knn", "svm", "softmax"]
if args.model == "knn":
    model = NearestNeighbor(args.cuda)
    model.train(train_datas, train_labels)

    # caculate forward time
    start = time.time()
    model.forward(test_datas, method=args.method)
    end = time.time()
    print('knn model forward time cost: {}'.format(end-start))

    # search K
    acc = list()
    for k in tqdm(range(1, args.maxk+1)):
        test_predictions = model.predict(k)
        acc.append(model.criterion(test_labels, test_predictions))
    start = time.time()
    print('knn model traverse k time cost: {}'.format(start-end))

    # draw image
    x = list(range(1, args.maxk+1))
    plt.plot(x, acc)
    plt.title('method={} maxk={} use_hog={}'.format(args.method, args.maxk, args.use_hog))
    plt.savefig('{}-{}-{}.png'.format(args.method, args.maxk, 'HOG' if args.use_hog else 'NO_HOG'))

    print('max accuracy: {}'.format(max(acc)))

elif args.model == "svm":
    model = SVM()

    # search learning_rates and regularization_strengths
    learning_rates = [1e-7, 1.3e-7, 1.4e-7, 1.5e-7, 1.6e-7]
    regularization_strengths = [8000.0, 9000.0, 10000.0, 11000.0, 18000.0, 19000.0, 20000.0]

    results = {}
    best_lr = None
    best_reg = None
    best_val = -1   # The highest validation accuracy that we have seen so far.
    best_svm = None # The LinearSVM object that achieved the highest validation rate.

    for lr in learning_rates:
        for reg in regularization_strengths:
            svm = SVM()

            loss_history = svm.train(train_datas, train_labels, learning_rate=lr, reg=reg, num_iters=args.iters, batch_size=200)
            y_train_pred = svm.predict(train_datas)
            accuracy_train = np.mean(y_train_pred == train_labels)

            y_val_pred = svm.predict(test_datas)
            accuracy_val = np.mean(y_val_pred == test_labels)

            if accuracy_val > best_val:
                best_lr = lr
                best_reg = reg
                best_val = accuracy_val
                best_svm = svm
            results[(lr, reg)] = accuracy_train, accuracy_val
            print('==> [Training SVM] lr: %e reg: %e train accuracy: %f val accuracy: %f' %
                    (lr, reg, results[(lr, reg)][0], results[(lr, reg)][1]))
    
    # finish training, report result
    print('==> [SVM] Best validation accuracy during cross-validation:\nlr = %e, reg = %e, best_val = %f' %
        (best_lr, best_reg, best_val))

elif args.model == "softmax":
    results = {}
    best_val = -1
    best_softmax = None
    learning_rates = [1e-6, 1e-7, 5e-7]
    regularization_strengths = [2.5e4, 5e4]

    for lr in tqdm(learning_rates):
        for rs in regularization_strengths:
            softmaxClassfier = softmax()
            loss_hist = softmaxClassfier.train(train_datas, train_labels, learning_rate=lr, reg=rs,
                        num_iters=args.iters, batch_size=args.batch_size, verbose=False)
            
            y_test_pred = softmaxClassfier.predict(test_datas)
            y_train_pred = softmaxClassfier.predict(train_datas)

            train_acc = np.mean(train_labels == y_train_pred)
            test_acc = np.mean(test_labels == y_test_pred)

            results[(lr, rs)] = (train_acc, test_acc)
            if test_acc > best_val:
                best_val = test_acc
                best_softmax = softmaxClassfier
    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                    lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during cross-validation: %f' % best_val)