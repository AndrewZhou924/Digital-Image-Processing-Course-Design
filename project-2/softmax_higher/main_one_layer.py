from cs231n.features import color_histogram_hsv, hog_feature
import matplotlib.pyplot as plt
from cs231n.data_utils import load_cifar10
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    # Load the raw CIFAR-10 data
    cifar10_dir = '../../cifar-10-batches-py'
    x_train, y_train, x_test, y_test = load_cifar10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    x_val = x_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    x_train = x_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    x_test = x_test[mask]
    y_test = y_test[mask]
    return x_train, y_train, x_val, y_val, x_test, y_test


x_train, y_train, x_val, y_val, x_test, y_test = get_CIFAR10_data()

from cs231n.features import *


num_color_bins = 10 # Number of bins in the color histogram
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
print(x_train.shape)
print(x_val.shape)


# x_train_feats = extract_features(x_train, feature_fns, verbose=True)
# x_val_feats = extract_features(x_val, feature_fns)
# x_test_feats = extract_features(x_test, feature_fns)

x_train_feats = x_train.reshape(-1, 3072)
x_val_feats = x_val.reshape(-1, 3072)
x_test_feats = x_test.reshape(-1, 3072)

# Preprocessing: Subtract the mean feature
#mean_feat = np.mean(x_train_feats, axis=0, keepdims=True)
#x_train_feats -= mean_feat
#x_val_feats -= mean_feat
#x_test_feats -= mean_feat

# Preprocessing: Divide by standard deviation. This ensures that each feature
# has roughly the same scale.
#std_feat = np.std(x_train_feats, axis=0, keepdims=True)
#x_train_feats /= std_feat
#x_val_feats /= std_feat
#x_test_feats /= std_feat

# # Preprocessing: Add a bias dimension
# X_train_feats = np.hstack([x_train_feats, np.ones((x_train_feats.shape[0], 1))])
# X_val_feats = np.hstack([x_val_feats, np.ones((x_val_feats.shape[0], 1))])
# X_test_feats = np.hstack([x_test_feats, np.ones((x_test_feats.shape[0], 1))])


print("x_train_feats", x_train_feats.shape)

from cs231n.classifiers.neural_net import  OneLayerNet
input_dim = x_train_feats.shape[1]
hidden_dim = 500
num_classes = 10

net = OneLayerNet(input_dim, hidden_dim, num_classes)
best_net = None
results = {}
best_val = -1
best_net = None

learning_rates = [1e-2, 1e-1, 5e-5, 1, 5]
regularization_strengths = [1e-3, 5e-3, 1e-2, 1e-1, 0.5, 1]

for lr in learning_rates:
    for reg in regularization_strengths:
        net = OneLayerNet(input_dim, hidden_dim, num_classes)
        #Train the network
        stats = net.train(x_train_feats, y_train, x_val_feats, y_val,
                          num_iters=3000, batch_size=200,
                          learning_rate=lr, learning_rate_decay=0.95,
                          reg=reg, verbose=False)

        val_acc = (net.predict(x_val_feats) == y_val).mean()
        if val_acc > best_val:
            best_val = val_acc
            best_net = net
        results[(lr, reg)] = val_acc

        plt.cla()
        plt.plot(stats['loss_history'])
        plt.xlabel('Iteration number')
        plt.ylabel('loss')
        info = 'loss_history_lr={}_re={}'.format(lr, reg)
        plt.title(info)
        plt.savefig("./plots/" + info + '.png')
        plt.close()

        plt.cla()
        plt.plot(stats['val_acc_history'])
        plt.xlabel('Iteration number')
        plt.ylabel('val_acc')
        info = 'val_acc_history_lr={}_re={}'.format(lr, reg)
        plt.title(info)
        plt.savefig("./plots/" + info + '.png')
        plt.close()

        plt.cla()
        plt.plot(stats['train_acc_history'])
        plt.xlabel('Iteration number')
        plt.ylabel('train_acc')
        info = 'train_acc_history_lr={}_re={}'.format(lr, reg)
        plt.title(info)
        plt.savefig("./plots/" + info + '.png')
        plt.close()

for lr, reg in sorted(results):
    val_acc = results[(lr, reg)]
    print('lr %e reg %e val accuracy:%f' % (lr, reg, val_acc))
print('best validation accuracy achieved during cross-validation:%f' % best_val)

test_acc = (best_net.predict(x_test_feats) == y_test).mean()
print(test_acc)
