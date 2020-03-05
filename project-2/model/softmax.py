#! -*-coding:utf8-*-
import numpy as np
# import cupy as np
from random import shuffle
from tqdm import tqdm 
from tools.gradient_check import grad_check,eval_numerical_gradient

class softmax(object):
    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        num_train, dim = X.shape
        num_classes = np.max(y) + 1

        # initialize weight metric W
        if self.W is None:
            # C :number of classes
            # D: dimension of each flattened image
            C, D = num_classes, 3072 
            self.W = np.random.randn(C, D) * 0.001

        loss_history = []
        for it in tqdm(range(num_iters)):
            X_batch = None
            y_batch = None
            
            # TODO add option

            # Batch
            # for batchId in range(int(num_train / batch_size)):
            #     X_batch = X[batchId*batch_size: (batchId+1)*batch_size]
            #     y_batch = y[batchId*batch_size: (batchId+1)*batch_size]
            #     loss, grad = self.loss(X_batch, y_batch, reg)
            #     loss_history.append(loss)
            #     self.W -= learning_rate * grad

            idx = np.random.choice(num_train, batch_size)
            X_batch = X[idx, :]
            y_batch = y[idx]
            X_batch = X_batch.T

            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            self.W -= learning_rate * grad

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def loss(self, x, y, reg):
        """
        Calculate the cross-entropy loss and the gradient for each iteration of training.
        Arguments:
            x: D * N numpy array as the training data, where D is the dimension and N the training sample size
            y: 1D numpy array with length N as the labels for the training data
        Output:
            loss: a float number of calculated cross-entropy loss
            dW: C * D numpy array as the calculated gradient for W, where C is the number of classes, and 10 for this model
        """

        # Calculation of loss
        z = np.dot(self.W, x)
        # Max trick for the softmax, preventing infinite values
        z -= np.max(z, axis=0)  
        # Softmax function
        p = np.exp(z) / np.sum(np.exp(z), axis=0)  
        # Cross-entropy loss
        L = -1 / len(y) * np.sum(np.log(p[y, range(len(y))]))  
        # Regularization term
        R = 0.5 * np.sum(np.multiply(self.W, self.W))  
        # Total loss
        loss = L + R * reg  

        # Calculation of dW
        p[y, range(len(y))] -= 1
        dW = 1 / len(y) * p.dot(x.T) + reg * self.W
        return loss, dW

    def predict(self, X):
        # y_pred = np.argmax(X.dot(self.W), axis=1)
        y_pred = np.argmax(np.dot(self.W, X.T), axis=0)
        return y_pred
  

  