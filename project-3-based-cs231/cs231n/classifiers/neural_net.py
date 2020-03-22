import numpy as np


class ThreeLayerNet(object):
    """
    A three-layer fully-connected neural network. The net has an input dimension of
    N, hidden layer dimension of H, another hidden layer of dimension H, and performs
    classification over C classes. We train the network with a softmax loss function
    and L2 regularization on the weight matrices. The network uses a ReLU nonlinearity
    after the first and second fully connected layer.
    In other words, the network has the following architecture:
    input - fully connected layer - ReLU - fully connected layer - ReLU - fully connected layer - softmax
    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-3):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:
        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, H)
        b2: Second layer biases; has shape (H,)
        W3: Third layer weights; has shape (H, C)
        b3: Third layer biases; has shape (C,)
        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a three layer fully connected neural
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
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # FCL + RELU
        first_layer = np.maximum(X.dot(W1)+b1,0)
        # FCL + RELU
        # np.maximum(,) takes two array to compare.
        second_layer = np.maximum(first_layer.dot(W2)+b2,0)
        scores = second_layer.dot(W3)+b3
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # First of all, as stated in lecture notes, we can shift them to the max.
        scores -= np.max(scores , axis=1).reshape(-1,1)
        divisor = np.sum(np.exp(scores) , axis=1).reshape(-1,1)

        # Our f is our softmax function and its output is "scores"
        softmax = np.exp(scores) / divisor
        loss = -np.sum(np.log(softmax[range(X.shape[0]), y]))
        loss = (loss / X.shape[0])+ (reg * (np.sum(W1*W1) + np.sum(W2*W2) +np.sum(W3*W3)))
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Let start to backward pass

        # 3rd layer
        dSoftmax = np.exp(scores)[np.arange(X.shape[0])] / np.sum(np.exp(scores),axis=1).reshape(-1,1)
        dSoftmax[np.arange(X.shape[0]),y] -= 1
        dLayer3 = dSoftmax/ X.shape[0]
        dW3 = np.transpose(second_layer).dot(dLayer3) + (2 * reg * W3)
        db3 = np.sum(dLayer3, axis=0)

        # 2nd layer
        dLayer2 = dLayer3.dot(np.transpose(W3))
        dLayer2[second_layer<=0]=0
        dW2 = np.transpose(first_layer).dot(dLayer2) + (2 * reg * W2)
        db2 = np.sum(dLayer2, axis=0)

        # Layer 1
        dLayer1 = dLayer2.dot(np.transpose(W2))
        dLayer1[first_layer<=0]=0
        dW1 = np.transpose(X).dot(dLayer1) + (2 * reg * W1)
        db1 = np.sum(dLayer1, axis=0)

        grads.update({'W1':dW1,'W2':dW2,'W3':dW3,'b1':db1,'b2':db2,'b3':db3});

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            random_indices = np.random.choice(X.shape[0], batch_size)
            X_batch, y_batch = X[random_indices] , y[random_indices]
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            def update_parameters(parameter_name):
                self.params[parameter_name] -= learning_rate * grads[parameter_name]

            parameters = ('W1', 'b1', 'W2','b2', 'W3', 'b3')
            for parameter in parameters:
                update_parameters(parameter)

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this three-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.
        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.
        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        def selfp(name):
            return self.params[name]

        # FC + Relu
        first_layer = np.maximum(X.dot(selfp("W1"))+selfp("b1"),0)
        # FC + Relu
        second_layer = np.maximum(first_layer.dot(selfp("W2"))+selfp("b2"),0)
        scores = second_layer.dot(selfp("W3"))+selfp("b3")
        # Max prediction
        y_pred = np.argmax(scores,axis=1)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
    
    
class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
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
        hidden_size = 50
        hidden1_size = 50
        
        self.params = dict()
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, hidden1_size)
        self.params['b2'] = np.zeros(hidden1_size)
        self.params['W3'] = std * np.random.randn(hidden1_size, output_size)
        self.params['b3'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a three layer fully connected neural
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
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # FCL + RELU
        first_layer = np.maximum(X.dot(W1)+b1,0)
        # FCL + RELU
        # np.maximum(,) takes two array to compare.
        second_layer = np.maximum(first_layer.dot(W2)+b2,0)
        scores = second_layer.dot(W3)+b3
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # First of all, as stated in lecture notes, we can shift them to the max.
        scores -= np.max(scores , axis=1).reshape(-1,1)
        divisor = np.sum(np.exp(scores) , axis=1).reshape(-1,1)

        # Our f is our softmax function and its output is "scores"
        softmax = np.exp(scores) / divisor
        loss = -np.sum(np.log(softmax[range(X.shape[0]), y]))
        loss = (loss / X.shape[0])+ (reg * (np.sum(W1*W1) + np.sum(W2*W2) +np.sum(W3*W3)))
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Let start to backward pass

        # 3rd layer
        dSoftmax = np.exp(scores)[np.arange(X.shape[0])] / np.sum(np.exp(scores),axis=1).reshape(-1,1)
        dSoftmax[np.arange(X.shape[0]),y] -= 1
        dLayer3 = dSoftmax/ X.shape[0]
        dW3 = np.transpose(second_layer).dot(dLayer3) + (2 * reg * W3)
        db3 = np.sum(dLayer3, axis=0)

        # 2nd layer
        dLayer2 = dLayer3.dot(np.transpose(W3))
        dLayer2[second_layer<=0]=0
        dW2 = np.transpose(first_layer).dot(dLayer2) + (2 * reg * W2)
        db2 = np.sum(dLayer2, axis=0)

        # Layer 1
        dLayer1 = dLayer2.dot(np.transpose(W2))
        dLayer1[first_layer<=0]=0
        dW1 = np.transpose(X).dot(dLayer1) + (2 * reg * W1)
        db1 = np.sum(dLayer1, axis=0)

        grads.update({'W1':dW1,'W2':dW2,'W3':dW3,'b1':db1,'b2':db2,'b3':db3});

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, grads
        
        
#     def loss(self, x, y=None, reg=0.0):
#         """
#         Compute the loss and gradients for a two layer fully connected neural
#         network.

#         Inputs:
#         - X: Input data of shape (N, D). Each X[i] is a training sample.
#         - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
#         an integer in the range 0 <= y[i] < C. This parameter is optional; if it
#         is not passed then we only return scores, and if it is passed then we
#         instead return the loss and gradients.
#         - reg: Regularization strength.

#         Returns:
#         If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
#         the score for class c on input X[i].

#         If y is not None, instead return a tuple of:
#         - loss: Loss (data loss and regularization loss) for this batch of training
#         samples.
#         - grads: Dictionary mapping parameter names to gradients of those parameters
#         with respect to the loss function; has the same keys as self.params.
#         """
#         # Unpack variables from the params dictionary
#         w1, b1 = self.params['W1'], self.params['b1']
#         w2, b2 = self.params['W2'], self.params['b2']
#         w3, b3 = self.params['W3'], self.params['b3']
#         n, d = x.shape

#         # Compute the forward pass
#         #############################################################################
#         # TODO:                                                                     #
#         # Perform the forward pass, computing the class scores for the input.       #
#         # Store the result in the scores variable, which should be an array of      #
#         # shape (N, C).                                                             #
#         #############################################################################

#         a1 = x.dot(w1) + self.params['b1']            # shape N x H
#         a1_relu = np.maximum(a1, np.zeros_like(a1))   # shape N x H
#         a2 = a1_relu.dot(w2) + self.params['b2']      # shape N x H1
#         a2_relu = np.maximum(a2, np.zeros_like(a2))   # shape N x H1
#         scores = a2_relu.dot(w3) + self.params['b3']  # shape N x C
#         #############################################################################
#         #                              END OF YOUR CODE                             #
#         #############################################################################

#         # If the targets are not given then jump out, we're done
#         if y is None:
#             return scores

#         # Compute the loss
#         #############################################################################
#         # TODO:                                                                     #
#         # Finish the forward pass, and compute the loss. This should include        #
#         # both the data loss and L2 regularization for W1 and W2. Store the result  #
#         # in the variable loss, which should be a scalar. Use the Softmax           #
#         # classifier loss.                                                          #
#         #############################################################################
#         # select in each row i the score at position y[i]
#         # the formula is given here http://cs231n.github.io/linear-classify/#softmax
#         correct_class_scores = scores[range(x.shape[0]), y].reshape(-1, 1)
#         # applying the log-sum-exp trick
#         # https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
#         max_scores = scores.max(axis=1, keepdims=True)
#         scores -= max_scores
#         # compute softmax loss
#         loss = - correct_class_scores.sum() + max_scores.sum() + np.log(np.exp(scores).sum(axis=1)).sum()
#         loss /= n
#         loss += reg * (np.sum(w1 * w1) + np.sum(w2 * w2) + np.sum(w3 * w3)) 
#         #############################################################################
#         #                              END OF YOUR CODE                             #
#         #############################################################################

#         # Backward pass: compute gradients
#         grads = {}
#         #############################################################################
#         # TODO:                                                                     #
#         # Compute the backward pass, computing the derivatives of the weights       #
#         # and biases. Store the results in the grads dictionary. For example,       #
#         # grads['W1'] should store the gradient on W1, and be a matrix of same size #
#         #############################################################################
#         softmax_deriv = (np.exp(scores) / np.exp(scores).sum(axis=1).reshape(-1, 1))
#         softmax_deriv[range(n), y] -= 1

#         # gradients of loss w.r.t. to weights W3
#         # study thoroughly the linear backprop example to understand the following
#         # http://cs231n.stanford.edu/handouts/linear-backprop.pdf
#         dw3 = a2_relu.T.dot(softmax_deriv)
#         dw3 /= n
#         dw3 += 2 * reg * w3
#         grads['W3'] = dw3

#         # gradients of loss w.r.t. to biases b3
#         db3 = np.sum(softmax_deriv, axis=0)
#         db3 /= n
#         grads['b3'] = db3

#         # upflow gradient after ReLU
#         da2_relu = softmax_deriv.dot(w3.T)

#         # gradients w.r.t. activation A1 after the first layer
#         da2 = da2_relu * (a2_relu > 0)

#         # gradients of loss w.r.t. to weights W2
#         dw2 = a1_relu.T.dot(da2)
#         dw2 /= n
#         dw2 += 2 * reg * w2
#         grads['W2'] = dw2

#         # gradients of loss w.r.t. to biases b2
#         db2 = np.sum(da2, axis=0)
#         db2 /= n
#         grads['b2'] = db2
        
        
#         # upflow gradient after ReLU
#         da1_relu = da2.dot(w2.T)
#         #da1_relu = softmax_deriv.dot(w3.T).dot(w2.T)

#         # gradients w.r.t. activation A1 after the first layer
#         da1 = da1_relu * (a1_relu > 0) #* (a2_relu > 0)

#         # gradients of loss w.r.t. to weights W2
#         dw1 = x.T.dot(da1)
#         dw1 /= n
#         dw1 += 2 * reg * w1
#         grads['W1'] = dw1

#         # gradients of loss w.r.t. to biases b2
#         db1 = np.sum(da1, axis=0)
#         db1 /= n
#         grads['b1'] = db1
        
#         #############################################################################
#         #                              END OF YOUR CODE                             #
#         #############################################################################

#         return loss, grads

    def train(self, x, y, x_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = x.shape[0]
        iterations_per_epoch = max(round(num_train / batch_size), 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            indexes = np.random.choice(x.shape[0], batch_size, replace=False)
            x_batch = x[indexes]
            y_batch = y[indexes]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(x_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO:                                                                 #
            # Use the gradients in the grads dictionary to update the               #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            for param_name in self.params:
                self.params[param_name] -= learning_rate * grads[param_name]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(x_batch) == y_batch).astype('int').mean()
                val_acc = (self.predict(x_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, x):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        scores = self.loss(x)
        y_pred = np.argmax(scores, axis=1)
        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################

        return y_pred

    

class ThreeMaxLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
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
        #hidden_size = 
        #hidden1_size = hidden_size
        
        self.params = dict()
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(input_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

    def loss(self, x, y=None, reg=0.0):
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
        grads = {}
        # Unpack variables from the params dictionary
        w1, b1 = self.params['W1'], self.params['b1']
        w2, b2 = self.params['W2'], self.params['b2']
        w3, b3 = self.params['W3'], self.params['b3']
        n, d = x.shape
        num_examples = x.shape[0]

        # Compute the forward pass
        #############################################################################
        # TODO:                                                                     #
        # Perform the forward pass, computing the class scores for the input.       #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        hidden_layer = np.maximum(x.dot(w1) + self.params['b1'], x.dot(w2) + self.params['b2']) 
        scores = hidden_layer.dot(w3) + b3
        
        
        if y is None:
            return scores
        
         # compute the class probabilities
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

        # compute the loss: average cross-entropy loss and regularization
        corect_logprobs = -np.log(probs[range(num_examples),y]).reshape(-1, 1)
        data_loss = np.sum(corect_logprobs)/num_examples
        reg_loss = 0.5*reg*np.sum(w1*w1) + 0.5*reg*np.sum(w2*w2) + 0.5*reg*np.sum(w3*w3)
        loss = data_loss + reg_loss
        #if i % 1000 == 0:
            #print "iteration %d: loss %f" % (i, loss)

        # compute the gradient on scores
        dscores = probs
        dscores[range(num_examples),y] -= 1
        dscores /= num_examples
        
        grads['W3'] = np.dot(hidden_layer.T, dscores)/n
        grads['b3'] = np.sum(dscores, axis=0, keepdims=True)/n
        
        #dhidden[hidden_layer <= 0] = 0
        dhidden2 = np.where(np.dot(x, w2) + b2 >= np.dot(x, w1) + b1, np.dot(dscores, w3.T), 0)
        dhidden  = np.where(np.dot(x, w2) + b2 <= np.dot(x, w1) + b1, np.dot(dscores, w3.T), 0)
  
        # finally into W,b, W2,b2
  
        grads['W2'] = np.dot(x.T, dhidden2)/n
        grads['b2'] = np.sum(dhidden2, axis=0, keepdims=True)/n
        grads['W1'] = np.dot(x.T, dhidden)/n
        grads['b1'] = np.sum(dhidden, axis=0, keepdims=True)/n
        

        # add regularization gradient contribution
        grads['W3'] += reg * w3
        grads['W2'] += reg * w2
        grads['W1'] += reg * w1
        
#         a1 = x.dot(w1) + self.params['b1']            # shape N x H
#         a1_relu = np.maximum(a1, np.zeros_like(a1))   # shape N x H
#         a2 = a1_relu.dot(w2) + self.params['b2']      # shape N x H1
#         a2_relu = np.maximum(a2, np.zeros_like(a2))   # shape N x H1
#         scores = a2_relu.dot(w3) + self.params['b3']  # shape N x C
#         #############################################################################
#         #                              END OF YOUR CODE                             #
#         #############################################################################

#         # If the targets are not given then jump out, we're done
#         if y is None:
#             return scores

#         # Compute the loss
#         #############################################################################
#         # TODO:                                                                     #
#         # Finish the forward pass, and compute the loss. This should include        #
#         # both the data loss and L2 regularization for W1 and W2. Store the result  #
#         # in the variable loss, which should be a scalar. Use the Softmax           #
#         # classifier loss.                                                          #
#         #############################################################################
#         # select in each row i the score at position y[i]
#         # the formula is given here http://cs231n.github.io/linear-classify/#softmax
#         correct_class_scores = scores[range(x.shape[0]), y].reshape(-1, 1)
#         # applying the log-sum-exp trick
#         # https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
#         max_scores = scores.max(axis=1, keepdims=True)
#         scores -= max_scores
#         # compute softmax loss
#         loss = - correct_class_scores.sum() + max_scores.sum() + np.log(np.exp(scores).sum(axis=1)).sum()
#         loss /= n
#         loss += reg * (np.sum(w1 * w1) + np.sum(w2 * w2) + np.sum(w3 * w3)) 
#         #############################################################################
#         #                              END OF YOUR CODE                             #
#         #############################################################################

#         # Backward pass: compute gradients
#         grads = {}
#         #############################################################################
#         # TODO:                                                                     #
#         # Compute the backward pass, computing the derivatives of the weights       #
#         # and biases. Store the results in the grads dictionary. For example,       #
#         # grads['W1'] should store the gradient on W1, and be a matrix of same size #
#         #############################################################################
#         softmax_deriv = (np.exp(scores) / np.exp(scores).sum(axis=1).reshape(-1, 1))
#         softmax_deriv[range(n), y] -= 1

#         # gradients of loss w.r.t. to weights W3
#         # study thoroughly the linear backprop example to understand the following
#         # http://cs231n.stanford.edu/handouts/linear-backprop.pdf
#         dw3 = a2_relu.T.dot(softmax_deriv)
#         dw3 /= n
#         dw3 += 2 * reg * w3
#         grads['W3'] = dw3

#         # gradients of loss w.r.t. to biases b3
#         db3 = np.sum(softmax_deriv, axis=0)
#         db3 /= n
#         grads['b3'] = db3

#         # upflow gradient after ReLU
#         da2_relu = softmax_deriv.dot(w3.T)

#         # gradients w.r.t. activation A1 after the first layer
#         da2 = da2_relu * (a2_relu > 0)

#         # gradients of loss w.r.t. to weights W2
#         dw2 = a1_relu.T.dot(da2)
#         dw2 /= n
#         dw2 += 2 * reg * w2
#         grads['W2'] = dw2

#         # gradients of loss w.r.t. to biases b2
#         db2 = np.sum(da2, axis=0)
#         db2 /= n
#         grads['b2'] = db2
        
        
#         # upflow gradient after ReLU
#         da1_relu = da2.dot(w2.T)
#         #da1_relu = softmax_deriv.dot(w3.T).dot(w2.T)

#         # gradients w.r.t. activation A1 after the first layer
#         da1 = da1_relu * (a1_relu > 0) #* (a2_relu > 0)

#         # gradients of loss w.r.t. to weights W2
#         dw1 = x.T.dot(da1)
#         dw1 /= n
#         dw1 += 2 * reg * w1
#         grads['W1'] = dw1

#         # gradients of loss w.r.t. to biases b2
#         db1 = np.sum(da1, axis=0)
#         db1 /= n
#         grads['b1'] = db1
        
#         #############################################################################
#         #                              END OF YOUR CODE                             #
#         #############################################################################

        return loss, grads

    def train(self, x, y, x_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = x.shape[0]
        iterations_per_epoch = max(round(num_train / batch_size), 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            indexes = np.random.choice(x.shape[0], batch_size, replace=False)
            x_batch = x[indexes]
            y_batch = y[indexes]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(x_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO:                                                                 #
            # Use the gradients in the grads dictionary to update the               #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            for param_name in self.params:
                #print(param_name, self.params[param_name].shape, self.params[param_name].shape)
                self.params[param_name] -= learning_rate * self.params[param_name]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(x_batch) == y_batch).astype('int').mean()
                val_acc = (self.predict(x_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, x):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        scores = self.loss(x)
        y_pred = np.argmax(scores, axis=1)
        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################

        return y_pred
