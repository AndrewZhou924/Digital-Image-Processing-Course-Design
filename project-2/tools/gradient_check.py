import numpy as np
from random import randrange

def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """
    fx = f(x)        # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h        # increment by h
        fxph = f(x)               # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)               # evaluate f(x - h)
        x[ix] = oldval            # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h)    # the slope
        if verbose:
            print(ix, grad[ix])
        it.iternext()          # step to next dimension
    return grad

def grad_check(f, x, analytic_grad, num_check_pairs):
    """
    Sample a few random elements and check the difference of numerical and analytical gradients.
    Arguments:
        f: a function that takes a single argument
        x: the point (numpy array) to evaluate the gradient at
        analytic_grad: Calculated gradient using analytical method
        num_check_pairs: Number of analytical-numerical gradient pairs you want to check
    """
    h = 1e-5

    for i in range(num_check_pairs):
        # Here the numerical gradients are calculated using df/dx = (f(x+h)+f(x-h))/2h
        ix = tuple([randrange(m) for m in x.shape])
        x[ix] += h  # Increment by h at dimension index ix
        fxph = f(x)  # Evaluate f(x + h)
        x[ix] -= 2 * h  # Decrement by h at dimension index ix
        fxmh = f(x)  # Evaluate f(x - h)
        x[ix] += h  # Reset x

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic  = analytic_grad[ix]
        relative_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        print('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, relative_error))

