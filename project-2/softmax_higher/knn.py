import numpy as np
from data_utils import load_cifar10
import matplotlib.pyplot as plt
from classification.knn import KNearestNeighbor
x_train, y_train, x_test, y_test = load_cifar10('cifar-10-batches-py')

print('training data shape:',x_train.shape)
print('training labels shape:',y_train.shape)
print('test data shape:',x_test.shape)
print('test labels shape:',y_test.shape)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_claesses = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace = False)
    for i, idx in enumerate(idxs):
        plt_idx = i*num_claesses + y + 1
        plt.subplot(samples_per_class, num_claesses, plt_idx)
        plt.imshow(x_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)

plt.show()

num_training = 5000
mask = range(num_training)
x_train = x_train[mask]
y_train = y_train[mask]
num_test = 500
mask = range(num_test)
x_test = x_test[mask]
y_test = y_test[mask]

x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))
print(x_train.shape, x_test.shape)

classifier = KNearestNeighbor()
classifier.train(x_train, y_train)
dists = classifier.compute_distances_two_loops(x_test)
print(dists)

y_test_pred = classifier.predict_labels(dists, k = 1)

num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

'''
在机器学习中，当数据量不是很充足时，交叉验证是一种不错的模型选择方法
（深度学习数据量要求很大，一般是不采用交叉验证的，因为它太费时间），
本节我们就利用交叉验证来选择最好的k值来获得较好的预测的准确率。

这里，我们采用S折交叉验证的方法，即将数据平均分成S份，
一份作为测试集，其余作为训练集，一般S=10，本文将S设为5，即代码中的num_folds=5。
'''

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
x_train_folds = []
y_train_folds = []

y_train = y_train.reshape(-1, 1)
x_train_folds = np.array_split(x_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

k_to_accuracies = {}

for k in k_choices:
    k_to_accuracies.setdefault(k, [])
for i in range(num_folds):
    classifier = KNearestNeighbor()
    x_val_train = np.vstack(x_train_folds[0:i]+x_train_folds[i+1:])
    y_val_train = np.vstack(y_train_folds[0:i]+y_train_folds[i+1:])
    y_val_train = y_val_train[:, 0]
    classifier.train(x_val_train, y_val_train)
    for k in k_choices:
        y_val_pred = classifier.predict(x_train_folds[i], k=k)  # 3.2
        num_correct = np.sum(y_val_pred == y_train_folds[i][:, 0])
        accuracy = float(num_correct) / len(y_val_pred)
        k_to_accuracies[k] = k_to_accuracies[k] + [accuracy]

for k in sorted(k_to_accuracies):
    sum_accuracy = 0
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))
        sum_accuracy+=accuracy
    print('the average accuracy is :%f' % (sum_accuracy / 5))

'''为了更形象的表示准确率，
   我们借助matplotlib.pyplot.
   errorbar函数来均值和偏差对应的趋势线'''
for k in k_choices:
    accuracies=k_to_accuracies[k]
    plt.scatter([k]*len(accuracies),accuracies)

accuracies_mean=np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std=np.array([np.std(v) for k ,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices,accuracies_mean,yerr=accuracies_std)
plt.title('cross-validation on k')
plt.xlabel('k')
plt.ylabel('cross-validation accuracy')
plt.show()
