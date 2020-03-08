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
from tools.gradient_check import grad_check,eval_numerical_gradient
import matplotlib.pyplot as plt
from PIL import Image
import random

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
parser.add_argument('--standardization', required=False, default=False, type=str2bool, help='data standardization')
parser.add_argument('--augumentation', required=False, default=False, type=str2bool, help='data augumentation')
parser.add_argument('--normalization', required=False, default=False, type=str2bool, help='data normalization')
parser.add_argument('--verbose', required=False, default=False, type=str2bool, help='verbose')
parser.add_argument('--checkGradient', required=False, default=False, type=str2bool, help='check Gradient')
parser.add_argument('--visualizeWeight', required=False, default=False, type=str2bool, help='visualize weight for each class')
parser.add_argument('--plotLoss', required=False, default=False, type=str2bool, help='visualize weight for each class')
parser.add_argument('--testMode', required=False, default=False, type=str2bool, help='testMode: only train once')
parser.add_argument('--plot_L_R_loss', required=False, default=False, type=str2bool, help='visual L and R loss seperately')

args = parser.parse_args()
if args.cuda:
    print("using cuda")
    import cupy as np
else:
    import numpy as np

def normalization(data):
   _range = np.max(data, axis=0) - np.min(data, axis=0)
   return (data - np.min(data, axis=0)) / _range
# def normalization(data):
#     feature_dim = int(data.shape[1] / 3)
#     data[:, 0: feature_dim] = (data[:, 0: feature_dim] - 0.4914) / 0.247
#     data[:, feature_dim: feature_dim*2] = (data[:, feature_dim: feature_dim*2] - 0.4822) / 0.243
#     data[:, feature_dim * 2: feature_dim * 3] = (data[:, feature_dim * 2: feature_dim * 3]- 0.4465) / 0.261
#     return data

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def flip_left_to_right(img):
    im = Image.fromarray(img, mode='RGB')
    out = im.transpose(Image.FLIP_LEFT_RIGHT)
    return np.array(out)

def flip_top_to_down(img):
    im = Image.fromarray(img, mode='RGB')
    out = im.transpose(Image.FLIP_TOP_BOTTOM)
    return np.array(out)

def add_black(img):
    x = random.randint(0, 27)
    y = x = random.randint(0, 27)
    img[:, :,  x: x + 4] = 0
    img[:, y: y + 4, :] = 0
    return np.array(img)

def augumentation(data, labels):
    data = numpy.concatenate(data).reshape((-1, 3, 32, 32))
    #data = numpy.transpose(data, (0, 2, 3, 1))

    new_data = []
    new_label = []
    flip_top_to_down_weight = 0.2
    flip_left_to_right_weight = 0.2
    black_weight = 0.2

    for i in range(len(data)):
        seed = random.random()
        new_data.append(data[i])
        new_label.append(labels[i])

        if seed < black_weight:
            new_data.append(add_black(data[i]))
            new_label.append(labels[i])

    shuffle_seed = random.randint(0, 100)
    random.seed(shuffle_seed)
    random.shuffle(new_data)
    random.seed(shuffle_seed)
    random.shuffle(new_label)

    new_data = np.array(new_data)
    num_data = len(new_data)
    new_data = new_data.reshape(num_data, -1)
    return new_data, np.array(new_label)

def unpickle(file: str) -> dict:
    '''
    :param file: path to one cifar10 item
    :return: dict of one cifar10 item
    '''
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d

def cifar10(root: str, use_hog=True, data_normalization=False, data_standardization=False, data_augumentation=False):# -> (np.array, np.array, np.array, np.array):
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
            train_features.append(hog(train_data[i], orientations=9, pixels_per_cell=(4, 4), cells_per_block=(3, 3), multichannel=True))
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

    # TODO
    # if augumentation
    if data_augumentation:
        print("==> procced data augumentation")
        train_datas, train_labels = augumentation(train_datas, train_labels)
        test_datas, test_labels = augumentation(test_datas, test_labels)

    # normalize to 0-1
    if data_normalization:
        print("==> procced data normolization")
        train_datas = normalization(train_datas)
        test_datas  = normalization(test_datas)

    if data_standardization:
        print("==> procced data standardization")
        train_datas = standardization(train_datas)
        test_datas  = standardization(test_datas)


    return (train_datas, train_labels, test_datas, test_labels)

def data_check(datas, labels):
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


# train_datas, train_labels, test_datas, test_labels, _ = get_CIFAR10_data(args.dir)
train_datas, train_labels, test_datas, test_labels = cifar10(args.dir, use_hog=args.use_hog, \
    data_normalization=args.normalization, data_standardization=args.standardization, data_augumentation=args.augumentation)

print("train_datas[0]", train_datas[0].shape)
print("train_datas[1]", train_datas[1].shape)

# test normalization
# ======================================================================
# train_data_copy = train_datas.copy()
# train_data_normalized = normalization(train_data_copy)

# orgin_image = train_data_copy[0].reshape(32,32,3)
# # train_data_copy = np.reshape(train_data_copy, (-1, 3, 32, 32))
# # orgin_image = np.transpose(orgin_image, (1, 2, 0))
# plt.imshow(orgin_image, interpolation='nearest')
# plt.show()
# plt.cla()

# train_data_normalized = np.reshape(train_data_normalized, (-1, 3, 32, 32))
# normalized_image = np.transpose(train_data_normalized[0], (1, 2, 0))
# print(normalized_image)
# plt.imshow(normalized_image, interpolation='nearest')
# plt.show()
# plt.cla()
# ======================================================================

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

    # check folder exist or not
    pathToCheck = ["./plots", "./log"]
    for path in pathToCheck:
        if not os.path.exists(path):
            os.mkdir(path)

    # Grid Search in hype-parameter
    learning_rates = [1e-5, 1e-9]#[1e-6, 1e-8]
    regularization_strengths = [2.5e2, 4e5]#[2.5e4, 4e4]
    # regularization_strengths = [10e2, 10e4]
    interval = 5

    # for lr in tqdm(learning_rates):
    for lr in tqdm(np.linspace(learning_rates[0], learning_rates[1], num=interval)):
        for rs in np.linspace(regularization_strengths[0], regularization_strengths[1], num=interval):
        # for rs in regularization_strengths:

            softmaxClassfier = softmax()
            loss_hist = softmaxClassfier.train(train_datas, train_labels, test_datas, test_labels, learning_rate=lr, reg=rs,
                        num_iters=args.iters, batch_size=args.batch_size, verbose=args.verbose)
            
            y_test_pred = softmaxClassfier.predict(test_datas)
            y_train_pred = softmaxClassfier.predict(train_datas)

            train_acc = np.mean(train_labels == y_train_pred)
            test_acc = np.mean(test_labels == y_test_pred)

            results[(lr, rs)] = (train_acc, test_acc)

            if test_acc > best_val:
                best_val = test_acc
                best_softmax = softmaxClassfier
            
            # Plot the loss for the training
            if args.plotLoss:
                plt.cla()
                plt.plot(loss_hist)
                plt.xlabel('Iteration number')
                plt.ylabel('Loss value')
                info = 'LossFigure_acc={} lr={} rs={} use_hog={} use_argu={} use_norm={} use_stand={}'.format(test_acc, \
                                        lr, rs, args.use_hog, args.augumentation, args.normalization, standardization)
                plt.title(info)
                plt.savefig("./plots/" + info + '.png')
                # plt.show()
                plt.close()

                # Plot the loss for the val
            if args.plotLoss:
                plt.cla()
                print("debug for val", softmaxClassfier.val_loss_history)
                plt.plot(softmaxClassfier.val_loss_history)
                plt.xlabel('Iteration number')
                plt.ylabel('Loss value')
                info = 'val_LossFigure_acc={} lr={} rs={} use_hog={} use_argu={} use_norm={} use_stand={}'.format(
                    test_acc, lr, rs, args.use_hog, args.augumentation, args.normalization, standardization)
                plt.title(info)
                plt.savefig("./plots/" + info + '.png')
                # plt.show()
                plt.close()

        #Plot the loss of L and R seperately
            if args.plot_L_R_loss:
                plt.cla()
                plt.plot(softmaxClassfier.loss_R_history)
                plt.xlabel('Iteration number')
                plt.ylabel('Loss value')
                info = 'R_LossFigure_acc={} lr={} rs={} use_hog={}'.format(test_acc, lr, rs, args.use_hog)
                plt.title(info)
                plt.savefig("./plots/" + info + '.png')

                plt.cla()
                plt.plot(softmaxClassfier.loss_L_history)
                plt.xlabel('Iteration number')
                plt.ylabel('Loss value')
                info = 'L_LossFigure_acc={} lr={} rs={} use_hog={}'.format(test_acc, lr, rs, args.use_hog)
                plt.title(info)
                plt.savefig("./plots/" + info + '.png')
                # plt.show()
                plt.close()

            # gradient check
            if args.checkGradient:
                f = lambda w: softmaxClassfier.loss(train_datas.T, train_labels, 0.0)[0]
                print('\n==> Gradient Checking:')
                loss, grad = softmaxClassfier.loss(train_datas.T, train_labels, 1e-6)
                grad_check(f, softmaxClassfier.W, grad, 5)

            if args.visualizeWeight:
                plt.cla()
                w = softmaxClassfier.W[:, :]
                w = np.reshape(w, (-1, 3, 32, 32))
                w = np.transpose(w, (0, 2, 3, 1))
                w = w.reshape(10, 32, 32, 3)
                w_min, w_max = np.min(w), np.max(w)

                classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                for i in range(10):
                    plt.subplot(2, 5, i + 1)

                    # Rescale the weights to be between 0 and 255 for image representation
                    w_img = 255.0 * (w[i].squeeze() - w_min) / (w_max - w_min)

                    plt.imshow(w_img.astype('uint8'))
                    plt.axis('off')
                    plt.title(classes[i])

                info = 'WeightFigure_acc={} lr={} rs={} use_hog={}'.format(test_acc, lr, rs, args.use_hog)
                # plt.title(info)
                plt.savefig("./plots/" + info + '.png')
                # plt.show()
                plt.close()

            if args.testMode:
                break  

        if args.testMode:
            break

    print("\n\n ======  train result  ====== \n")
    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('softmax: lr %e reg %e train accuracy: %f val accuracy: %f' % (
                    lr, reg, train_accuracy, val_accuracy))
        
    print('[softmax] best validation accuracy achieved during cross-validation: %f' % best_val)
