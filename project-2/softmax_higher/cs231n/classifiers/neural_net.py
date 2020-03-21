import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size, std = 1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:
        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)
        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['w1'] = std*np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = std*np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, x, y = None, reg = 0.0):
        """
            Compute the loss and gradients for a two layer fully connected neural
            network.
            Inputs:
            - X: Input data of shape (N, D). Each X[i] is a training sample.
            - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
              an integer in the range 0 <= y[i] < C. This parameter is optional; if it
              is not passed then we only return scores, and if it is passed then we
              instead return the loss and gradients.
            - reg: Regularization strength.
            Returns:
            If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
            the score for class c on input X[i].
            If y is not None, instead return a tuple of:
            - loss: Loss (data loss and regularization loss) for this batch of training
              samples.
            - grads: Dictionary mapping parameter names to gradients of those parameters
              with respect to the loss function; has the same keys as self.params.
        """
        w1, b1 = self.params['w1'], self.params['b1']
        w2, b2 = self.params['w2'], self.params['b2']
        N, D = x.shape

        scores = None

        h_output = np.maximum(0, x.dot(w1) + b1) #Relu
        scores = h_output.dot(w2) + b2       #output


        if y is None:
            return scores
        loss = None
        #forward pass
        shift_scores = scores - np.max(scores, axis = 1).reshape(-1,1)  #reshape(-1, 1) reshape the array to one cols
        softmax_output = np.exp(shift_scores)/np.sum(np.exp(shift_scores), axis=1).reshape(-1, 1)
        loss = -np.sum(np.log(softmax_output[range(N), list(y)]))
        loss /= N
        loss += 0.5 * reg * (np.sum(w1) + np.sum(np.abs(w2)))

        #backward pass : compute gradients
        grads = {}

        #the second layer grad computer
        dscores = softmax_output.copy()
        dscores[range(N), list(y)] -= 1
        dscores /= N
        if w2 > 0:
            grads['w2'] = h_output.T.dot(dscores) + reg
        elif w2 < 0:
            grads['w2'] = h_output.T.dot(dscores) - reg
        else:
            grads['w2'] = h_output.T.dot(dscores)

        grads['b2'] = np.sum(dscores, axis = 0)
        #the first layer grad computer
        dh = dscores.dot(w2.T)
        dh_ReLu = (h_output > 0) * dh
        if w1 > 0:
            grads['w1'] = x.T.dot(dh_ReLu) + reg
        elif w1 < 0:
            grads['w1'] = x.T.dot(dh_ReLu) - reg
        else:
            grads['w1'] = x.T.dot(dh_ReLu)
        grads['b1'] = np.sum(dh_ReLu, axis = 0)
        return loss, grads

    def loss(self, X, y=None, reg=0.0):
        """
        计算两层全连接神经网络的loss和gradients
        输入：
        - X ：输入维数为（N,D）
        - y : 输入维数为（N，）
        - reg : 正则化强度
        返回：
        如果y是None,返回维数为(N,C)的分数矩阵
        如果y 不是None ,则返回一个元组：
        - loss : float 类型，数据损失和正则化损失
        - grads : 一个字典类型，存储W1，W2，b1,b2的梯度
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['w1'], self.params['b1']
        W2, b2 = self.params['w2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None

        # 完成前向传播，并且计算loss
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #

        h1 = np.maximum(0, np.dot(X, W1) + b1)
        h2 = np.dot(h1, W2) + b2
        scores = h2

        # If the targets are not given then jump out, we're done

        if y is None:
            return scores

        # Compute the loss
        loss = None

        exp_class_score = np.exp(scores)
        exp_correct_class_score = exp_class_score[np.arange(N), y]

        loss = -np.log(exp_correct_class_score / np.sum(exp_class_score, axis=1))
        loss = sum(loss) / N

        loss += reg * (np.sum(W2 ** 2) + np.sum(W1 ** 2))

        # Backward pass: compute gradients
        grads = {}

        # 计算反向传播，将权重和偏置量的梯度存储在params字典中
        # layer2
        dh2 = exp_class_score / np.sum(exp_class_score, axis=1, keepdims=True)
        dh2[np.arange(N), y] -= 1
        dh2 /= N

        dW2 = np.dot(h1.T, dh2)
        dW2 += 2 * reg * W2

        db2 = np.sum(dh2, axis=0)

        # layer1
        dh1 = np.dot(dh2, W2.T)

        dW1X_b1 = dh1
        dW1X_b1[h1 <= 0] = 0

        dW1 = np.dot(X.T, dW1X_b1)
        dW1 += 2 * reg * W1

        db1 = np.sum(dW1X_b1, axis=0)

        grads['w2'] = dW2
        grads['b2'] = db2
        grads['w1'] = dW1
        grads['b1'] = db1

        return loss, grads

    def train(self, x, y, x_val, y_val,
               learning_rate=1e-3, learning_rate_decay=0.95,
               reg=1e-5, num_iters=100,
               batch_size = 200, verbose = False):

        num_train = x.shape[0]
        iterations_per_epoch = max(num_train/batch_size, 1)

        #Use SGD to optimize the parameters in self.model

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            x_batch = None
            y_batch = None
            idx = np.random.choice(num_train, batch_size, replace = True)
            x_batch = x[idx]
            y_batch = y[idx]

            #compute loss and gradients using the current minibatch
            loss, grads = self.loss(x_batch, y = y_batch, reg = reg)
            loss_history.append(loss)

            #params update
            self.params['w2'] += -learning_rate * grads['w2']
            self.params['b2'] += -learning_rate * grads['b2']
            self.params['w1'] += -learning_rate * grads['w1']
            self.params['b1'] += -learning_rate * grads['b1']

            if verbose and (it % 100 == 0) :
                print('iteration %d / %d : loss %f' %(it, num_iters, loss))

            #every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                #check accuracy
                train_acc = (self.predict(x_batch) == y_batch).mean()
                val_acc = (self.predict(x_val) == y_val).mean()
                train_acc_history.append(val_acc)

                #decay learning rate
                learning_rate *= learning_rate_decay
        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history
        }

    def predict(self, x):
        y_pred = None
        h = np.maximum(0, x.dot(self.params['w1']) + self.params['b1'])
        scores = h.dot(self.params['w2'])+self.params['b2']
        y_pred = np.argmax(scores, axis = 1)
        return y_pred




class OneLayerNetWithRelu(object):
    def __init__(self, input_size, hidden_size, output_size, std = 1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['w2'] = std*np.random.randn(input_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, x, y = None, reg = 0.0):
        """
            Compute the loss and gradients for a two layer fully connected neural
            network.

            Inputs:
            - X: Input data of shape (N, D). Each X[i] is a training sample.
            - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
              an integer in the range 0 <= y[i] < C. This parameter is optional; if it
              is not passed then we only return scores, and if it is passed then we
              instead return the loss and gradients.
            - reg: Regularization strength.

            Returns:
            If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
            the score for class c on input X[i].

            If y is not None, instead return a tuple of:
            - loss: Loss (data loss and regularization loss) for this batch of training
              samples.
            - grads: Dictionary mapping parameter names to gradients of those parameters
              with respect to the loss function; has the same keys as self.params.
        """
        w2, b2 = self.params['w2'], self.params['b2']
        N, D = x.shape

        scores = None

        #h_output = np.maximum(0, x.dot(w2) + b2) #Relu
        scores = x.dot(w2) + b2
        h_output = np.maximum(0, scores) #Relu#output

        if y is None:
            return scores
        loss = None
        #forward pass
        shift_scores = scores - np.max(scores, axis = 1).reshape(-1,1)  #reshape(-1, 1) reshape the array to one cols
        softmax_output = np.exp(shift_scores)/np.sum(np.exp(shift_scores), axis=1).reshape(-1, 1)
        loss = -np.sum(np.log(softmax_output[range(N), list(y)]))
        loss /= N
        loss += 0.5 * reg * (np.sum(w2*w2))

        #backward pass : compute gradients
        grads = {}

        #the second layer grad computer
        dscores = softmax_output.copy()
        dscores[range(N), list(y)] -= 1
        dscores /= N

        dscores = (h_output > 0) *dscores


        grads['w2'] = x.T.dot(dscores) + reg * w2
        grads['b2'] = np.sum(dscores, axis = 0)

        return loss, grads


    def train(self, x, y, x_val, y_val,
               learning_rate=1e-3, learning_rate_decay=0.95,
               reg=1e-5, num_iters=100,
               batch_size = 200, verbose = False):

        num_train = x.shape[0]
        iterations_per_epoch = max(num_train/batch_size, 1)

        #Use SGD to optimize the parameters in self.model

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            x_batch = None
            y_batch = None
            idx = np.random.choice(num_train, batch_size, replace = True)
            x_batch = x[idx]
            y_batch = y[idx]

            #compute loss and gradients using the current minibatch
            loss, grads = self.loss(x_batch, y = y_batch, reg = reg)
            loss_history.append(loss)

            #params update
            self.params['w2'] += -learning_rate * grads['w2']
            self.params['b2'] += -learning_rate * grads['b2']

            if verbose and (it % 100 == 0) :
                print('iteration %d / %d : loss %f' %(it, num_iters, loss))

            #every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                #check accuracy
                train_acc = (self.predict(x_batch) == y_batch).mean()
                val_acc = (self.predict(x_val) == y_val).mean()
                train_acc_history.append(val_acc)

                #decay learning rate
                learning_rate *= learning_rate_decay
        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history
        }

    def predict(self, x):
        y_pred = None
        scores = x.dot(self.params['w2'])+self.params['b2']
        y_pred = np.argmax(scores, axis = 1)
        return y_pred







class OneLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size, std = 1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['w2'] = std*np.random.randn(input_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, x, y = None, reg = 0.0):
        """
            Compute the loss and gradients for a two layer fully connected neural
            network.

            Inputs:
            - X: Input data of shape (N, D). Each X[i] is a training sample.
            - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
              an integer in the range 0 <= y[i] < C. This parameter is optional; if it
              is not passed then we only return scores, and if it is passed then we
              instead return the loss and gradients.
            - reg: Regularization strength.

            Returns:
            If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
            the score for class c on input X[i].

            If y is not None, instead return a tuple of:
            - loss: Loss (data loss and regularization loss) for this batch of training
              samples.
            - grads: Dictionary mapping parameter names to gradients of those parameters
              with respect to the loss function; has the same keys as self.params.
        """
        w2, b2 = self.params['w2'], self.params['b2']
        N, D = x.shape

        scores = None

        #h_output = np.maximum(0, x.dot(w2) + b2) #Relu
        scores = x.dot(w2) + b2       #output


        if y is None:
            return scores
        loss = None
        #forward pass
        shift_scores = scores - np.max(scores, axis = 1).reshape(-1,1)  #reshape(-1, 1) reshape the array to one cols
        softmax_output = np.exp(shift_scores)/np.sum(np.exp(shift_scores), axis=1).reshape(-1, 1)
        loss = -np.sum(np.log(softmax_output[range(N), list(y)]))
        loss /= N
        loss += 0.5 * reg * (np.sum(w2*w2))

        #backward pass : compute gradients
        grads = {}

        #the second layer grad computer
        dscores = softmax_output.copy()
        dscores[range(N), list(y)] -= 1
        dscores /= N
        grads['w2'] = x.T.dot(dscores) + reg * w2
        grads['b2'] = np.sum(dscores, axis = 0)

        return loss, grads

    def loss(self, X, y=None, reg=0.0):
        """
        计算两层全连接神经网络的loss和gradients
        输入：
        - X ：输入维数为（N,D）
        - y : 输入维数为（N，）
        - reg : 正则化强度

        返回：
        如果y是None,返回维数为(N,C)的分数矩阵
        如果y 不是None ,则返回一个元组：
        - loss : float 类型，数据损失和正则化损失
        - grads : 一个字典类型，存储W1，W2，b1,b2的梯度
        """
        # Unpack variables from the params dictionary
        W2, b2 = self.params['w2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None

        # 完成前向传播，并且计算loss
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #

        h2 = np.dot(X, W2) + b2
        scores = h2

        # If the targets are not given then jump out, we're done

        if y is None:
            return scores

        # Compute the loss
        loss = None

        exp_class_score = np.exp(scores)
        exp_correct_class_score = exp_class_score[np.arange(N), y]

        loss = -np.log(exp_correct_class_score / np.sum(exp_class_score, axis=1))
        loss = sum(loss) / N

        loss += reg * (np.sum(W2 ** 2))

        # Backward pass: compute gradients
        grads = {}

        # 计算反向传播，将权重和偏置量的梯度存储在params字典中
        # layer2
        dh2 = exp_class_score / np.sum(exp_class_score, axis=1, keepdims=True)
        dh2[np.arange(N), y] -= 1
        dh2 /= N

        dW2 = np.dot(X.T, dh2)
        dW2 += 2 * reg * W2

        db2 = np.sum(dh2, axis=0)


        grads['w2'] = dW2
        grads['b2'] = db2

        return loss, grads

    def train(self, x, y, x_val, y_val,
               learning_rate=1e-3, learning_rate_decay=0.95,
               reg=1e-5, num_iters=100,
               batch_size = 200, verbose = False):

        num_train = x.shape[0]
        iterations_per_epoch = max(num_train/batch_size, 1)

        #Use SGD to optimize the parameters in self.model

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            x_batch = None
            y_batch = None
            idx = np.random.choice(num_train, batch_size, replace = True)
            x_batch = x[idx]
            y_batch = y[idx]

            #compute loss and gradients using the current minibatch
            loss, grads = self.loss(x_batch, y = y_batch, reg = reg)
            loss_history.append(loss)

            #params update
            self.params['w2'] += -learning_rate * grads['w2']
            self.params['b2'] += -learning_rate * grads['b2']

            if verbose and (it % 100 == 0) :
                print('iteration %d / %d : loss %f' %(it, num_iters, loss))

            #every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                #check accuracy
                train_acc = (self.predict(x_batch) == y_batch).mean()
                val_acc = (self.predict(x_val) == y_val).mean()
                train_acc_history.append(val_acc)

                #decay learning rate
                learning_rate *= learning_rate_decay
        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history
        }

    def predict(self, x):
        y_pred = None
        scores = x.dot(self.params['w2'])+self.params['b2']
        y_pred = np.argmax(scores, axis = 1)
        return y_pred




























































