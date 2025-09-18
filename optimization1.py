"""Logistic regression model"""

import numpy as np
from strategic1 import best_response


def sigmoid(z):
    """Evaluate sigmoid function"""
    return 1 / (1 + np.exp(-z))


def evaluate_loss(X, Y, theta, lam, strat_features=[], epsilon=0):
    """Evaluate L2-regularized logistic regression loss function. For epsilon>0 it returns the performative loss.

    Parameters
    ----------
        X: np.array
            training data matrix
        Y: np.array
            labels
        theta: np.array
            parameter vector
        lam: float
            regulariaation parameter, lam>0
        strat_features: list
            list of features that can be manipulated strategically, other features remain fixed
        epsilon: float
            sensitivity parameter, quantifying the strength of performative effects  

    Returns
    -------
        loss: float
            logistic loss value  
    """

    n = X.shape[0]

    # compute strategically manipulated data
    if epsilon > 0:
        X_perf = best_response(X, theta, epsilon, strat_features)
    else:
        X_perf = np.copy(X)

    # compute log likelihood
    t1 = 1.0/n * np.sum(-1.0 * np.multiply(Y, X_perf @ theta) +
                        np.log(1 + np.exp(X_perf @ theta)))

    # add regularization (without considering the bias)
    t2 = lam / 2.0 * np.linalg.norm(theta[:-1]) ** 2
    loss = t1 + t2

    return loss


def logistic_regression(X_orig, Y_orig, lam, z_init=None):
    """Training of an L2-regularized logistic regression model.

    Parameters
    ----------
        X_orig: np.array
            training data matrix
        Y_orig: np.array
            labels
        lam: float
            regulariation parameter, lam>0
        method: string
            optimization method: 'Exact' for returning the exact solution and 'GD' for performing a single gradient descent step on the parameter vector
        tol: float
            stopping criterion for exact minimization
        theta_init: np.array
            initial parameter vector. If None procedure is initialized at zero

    Returns
    -------
        theta: np.array
            updated parameter vector
        loss_list: list
            loss values furing training for reporting
        smoothness: float
            smoothness parameter of the logistic loss function given the current training data matrix
    """

    # assumes that the last coordinate is the bias term
    X = np.copy(X_orig)
    Y = np.copy(Y_orig)
    n, d = X.shape

    # compute smoothness of the logistic loss
    smoothness = np.sum(np.square(np.linalg.norm(X, axis=1))) / (4.0 * n)


    eta = 1 / (smoothness +  2*lam)

    if z_init is not None:
        z = np.copy(z_init)
    else:
        z = np.zeros(d)

    # evaluate initial loss
    prev_loss = evaluate_loss(X, Y, z, lam)

    # take gradients
    exp_tx = np.exp(X @ z)
    c = exp_tx / (1 + exp_tx) - Y
    gradient = 1.0/n * \
        np.sum(X * c[:, np.newaxis], axis=0) + \
        lam * np.append(z[:-1], 0)

    new_z = z - eta * gradient

    # compute new loss
    new_loss = evaluate_loss(X, Y, new_z, lam)


    z = np.copy(new_z)



    

    return z, new_loss, smoothness

