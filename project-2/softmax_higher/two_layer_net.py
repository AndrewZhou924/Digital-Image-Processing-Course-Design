from homework.cs231n.data_utils import load_cifar10
import numpy as np
from homework.cs231n.classifiers.neural_net import TwoLayerNet
import matplotlib.pyplot as plt


def get_cifar10_data(num_training = 49000, num_validation = 1000, num_test = 1000):
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    x_train, y_train, x_test, y_test = load_cifar10(cifar10_dir)

    #subsample the data
    mask = range(num_training, num_training + num_validation)
    x_val = x_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    x_train = x_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    x_test = x_test[mask]
    y_test = y_test[mask]

    #Normalize the data: subtract the mean image
    mean_image = np.mean(x_train, axis = 0)
    x_train -= mean_image
    x_val -= mean_image
    x_test -= mean_image

    #reshape data to rows
    x_train = x_train.reshape(num_training, -1)
    x_val = x_val.reshape(num_validation, -1)
    x_test = x_test.reshape(num_test, -1)

    return x_train, y_train, x_val, y_val, x_test, y_test

# Invoke the above function to get our data.
x_train, y_train, x_val, y_val, x_test, y_test = get_cifar10_data()
print('Train data shape: ', x_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', x_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', x_test.shape)
print('Test labels shape: ', y_test.shape)

input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)

#Train the network use SGD
stats = net.train(x_train, y_train, x_val, y_val,
                  num_iters = 1000, batch_size = 200,
                  learning_rate = 1e-4, learning_rate_decay = 0.95,
                  reg = 0.5, verbose = True)
#Predict on the validation set
val_acc = (net.predict(x_val) == y_val).mean()
print('validation accuracy: ', val_acc)

#Plot the loss function and train / validation accuracies
plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label = 'train')
plt.plot(stats['val_acc_history'], label = 'train')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.show()

from homework.cs231n.vis_utils import visualize_grid

# Visualize the weights of the network

def show_net_weights(net):
  W1 = net.params['w1']
  W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
  plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
  plt.gca().axis('off')
  plt.show()

show_net_weights(net)

input_size = 32*32*3
num_classes = 10
hidden_size = [75, 100, 125]
results = {}
best_val_acc = 0
best_net = None

learning_rates = np.array([0.7, 0.8, 0.9, 1.0, 1.1]) * 1e-3
regularization_strengths = [0.75, 1.9, 1.25]
print('running')
for hs in hidden_size:
    for lr in learning_rates:
        for reg in regularization_strengths:
            net = TwoLayerNet(input_size, hs, num_classes)

            stats = net.train(x_train, y_train, x_val, y_val,
                              num_iters = 1500, batch_size = 200,
                              learning_rate = lr, learning_rate_decay = 0.95,
                              reg = reg, verbose = False)
            val_acc = (net.predict(x_val) == y_val).mean()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_net = net
            results[(hs, lr, reg)] = val_acc

print('finshed')
for hs, lr, reg in sorted(results):
    val_acc = results[(hs, lr, reg)]
    print('hs %d lr %e reg %e val accuracy: %f' % (hs, lr, reg, val_acc))

print('best validation accuracy achieved during cross_validation: %f' % best_val_acc)
test_acc = (best_net.predict(x_test) == y_test).mean()
print('test_accuracy:', test_acc)

