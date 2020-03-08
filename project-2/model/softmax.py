#! -*-coding:utf8-*-
import numpy as np
# import cupy as np
from random import shuffle
from tqdm import tqdm 
from tools.gradient_check import grad_check,eval_numerical_gradient

class softmax(object):
    def __init__(self):
        self.W = None
        self.loss_L_history = []
        self.loss_R_history = []
        self.val_loss_history = []

    def train(self, X, y, val_X, val_y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        num_train, dim = X.shape
        num_val, val_dim = val_X.shape
        num_classes = np.max(y) + 1

        # initialize weight metric W
        if self.W is None:
            # C :number of classes
            # D: dimension of each flattened image
            C, D = num_classes, 3072  #use_hog should be 324
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

            # idx = np.random.choice(num_val, batch_size)
            # val_X_batch = val_X[idx, :]
            # val_y_batch = val_y[idx]
            # val_X_batch = val_X_batch.T
            #
            #
            # val_loss, val_grad = self.loss(val_X_batch, val_y_batch, reg)
            # self.val_loss_history.append(val_loss)

            #print("train", y_batch.shape)
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            self.W -= learning_rate * grad

            if it in [2000, 6000]:
                learning_rate *= 0.1

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
        if np.sum(np.log(p[y, range(len(y))]))  == 0:

            print("len(y)", len(y))
            print(" np.sum(np.log(p[y, range(len(y))])) ",  np.sum(np.log(p[y, range(len(y))])) )
        L = -1 / len(y) * np.sum(np.log(p[y, range(len(y))]))  
        # Regularization term
        R = 0.5 * np.sum(np.multiply(self.W, self.W))  
        # Total loss
        loss = L + R * reg
        self.loss_L_history.append(L)
        self.loss_R_history.append(R)
        # Calculation of dW
        p[y, range(len(y))] -= 1
        dW = 1 / len(y) * p.dot(x.T) + reg * self.W
        return loss, dW

    def predict(self, X):
        # y_pred = np.argmax(X.dot(self.W), axis=1)
        y_pred = np.argmax(np.dot(self.W, X.T), axis=0)
        return y_pred
  

  