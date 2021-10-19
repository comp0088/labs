#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
COMP0088 lab exercises for week 2.

Add your code as specified below.

A simple test driver is included in this script. Call it at the command line like this:

  $ python week_2.py

A 4-panel figure, `week_2.pdf`, will be generated so you can check it's doing what you
want. You should not need to edit the driver code, though you can if you wish.

NB: the code imports two functions from last week's exercises. If you were not
able to complete those functions, please talk to the TAs to get a working version.
"""

import sys, os, os.path
import argparse

import numpy as np
import numpy.random
import matplotlib.pyplot as plt

import utils

from week_1_working import generate_noisy_linear, generate_linearly_separable

#### ADD YOUR CODE BELOW

# -- Question 1 --

def ridge_closed ( X, y, l2=0 ):
    """
    Implement L2-penalised least-squares (ridge) regression
    using its closed form expression.
    
    # Arguments
        X: an array of sample data, where rows are samples
           and columns are features (assume there are at least
           as many samples as features). caller is responsible
           for prepending x0=1 terms if required.
        y: vector of measured (or simulated) labels for the samples,
           must be same length as number of rows in X
        l2: optional L2 regularisation weight. if zero (the default)
           then this reduces to unregularised least squares
    
    # Returns
        w: the fitted vector of weights  
    """
    assert(len(X.shape)==2)
    assert(X.shape[0]==len(y))
        
    return np.linalg.solve(X.T @ X + l2 * np.identity(X.shape[1]), X.T @ y)

    # other options might be:
    #
    #   return np.linalg.inv(X.T @ X + l2 * np.identity(X.shape[1])) @ X.T @ y
    #   return np.linalg.pinv(X.T @ X + l2 * np.identity(X.shape[1])) @ X.T @ y


# -- Question 2 --

def monomial_projection_1d ( X, degree ):
    """
    Map 1d data to an expanded basis of monomials
    up to the given degree.
    
    # Arguments
        X: an array of sample data, where rows are samples
            and the single column is the input feature.
        degree: maximum degree of the monomial terms
    
    # Returns
        Xm: an array of the transformed data, with the
            same number of rows (samples) as X, and
            with degree+1 columns (features):
            1, x, x**2, x**3, ..., x**degree
    """
    assert(len(X.shape)==2)
    assert(X.shape[1]==1)

    Xm = np.zeros((X.shape[0], degree+1))
    Xm[:,0] = 1
    Xm[:,1] = X[:,0]
    
    for ii in range(2, degree+1):
        Xm[:,ii] = X[:, 0] ** ii
    
    return Xm


def generate_noisy_poly_1d ( num_samples, weights, sigma, limits, rng ):
    """
    Draw samples from a 1D polynomial model with additive
    Gaussian noise.
    
    # Arguments
        num_samples: number of samples to generate
            (ie, the number of rows in the returned X
            and the length of the returned y)
        weights: vector of the polynomial coefficients
            (including a bias term at index 0)
        sigma: standard deviation of the additive noise
        limits: a tuple (low, high) specifying the value
            range for the single input dimension x1
        rng: an instance of numpy.random.Generator
            from which to draw random numbers
    
    # Returns
        X: a matrix of sample inputs, where
            the samples are the rows and the
            single column is the 1D feature x1
            ie, its size should be:
              num_samples x 1
        y: a vector of num_samples output values
    """
    return utils.random_sample(lambda x: utils.affine(monomial_projection_1d(x, len(weights)-1), weights),
                               1,
                               num_samples, limits, rng, sigma)

    
def fit_poly_1d ( X, y, degree, l2=0 ):
    """
    Fit a polynomial of the given degree to 1D sample data.
    
    # Arguments
        X: an array of sample data, where rows are samples
            and the single column is the input feature.
        y: vector of output values corresponding to the inputs,
           must be same length as number of rows in X
        degree: degree of the polynomial
        l2: optional L2 regularisation weight
    
    # Returns
        w: the fitted polynomial coefficients
    """
    assert(len(X.shape)==2)
    assert(X.shape[1]==1)
    assert(X.shape[0]==len(y))
    
    Xm = monomial_projection_1d(X, degree)
    return ridge_closed( Xm, y, l2 )
        

# -- Question 3 --

def gradient_descent ( z, loss_func, grad_func, lr=0.01,
                       loss_stop=1e-4, z_stop=1e-4, max_iter=100 ):
    """
    Generic batch gradient descent optimisation.
    Iteratively updates z by subtracting lr * grad
    until one or more stopping criteria are met.
    
    # Arguments
        z: initial value(s) of the optimisation var(s).
            can be a scalar if optimising a univariate
            function, otherwise a single numpy array
        loss_func: function of z that we seek to minimise,
            should return a scalar value
        grad_func: function calculating the gradient of
            loss_func at z. for vector z, this should return
            a vector of the same length containing the
            partial derivatives
        lr: learning rate, ie fraction of the gradient by 
            which to update z each iteration
        loss_stop: stop iterating if the loss changes
            by less than this (absolute)
        z_stop: stop iterating if z changes by less than
            this (L2 norm)
        max_iter: stop iterating after iterating this
            many times
    
    # Returns
        zs: a list of the z values at each iteration
        losses: a list of the losses at each iteration
    """
    losses = [ loss_func(z) ]
    zs = [ z ]
    
    d_loss = np.inf
    d_z = np.inf

    while (len(losses) <= max_iter) and (d_loss > loss_stop) and (d_z > z_stop):
        zs.append(zs[-1] - lr * grad_func(zs[-1]))
        losses.append(loss_func(zs[-1]))
        
        d_loss = np.abs(losses[-2] - losses[-1])
        d_z = np.linalg.norm(zs[-2] - zs[-1])
    
    return zs[1:], losses[1:]



# -- Question 3 --

# auxiliary functions
def sigmoid ( z ):
    return 1/(1 + np.exp(-z))

def logistic_forward ( X, w ):
    return sigmoid(utils.affine(X, w))

def logistic_loss ( X, y, w, eps=1e-10 ):
    g = logistic_forward(X, w)
    return (np.dot(-y, np.log(g + eps)) - np.dot((1 - y), np.log(1 - g + eps)))/len(y)

def logistic_grad ( X, y, w ):
    g = logistic_forward(X, w)
    return X.T @ (g - y)


def logistic_regression ( X, y, w0=None, lr=0.05,
                          loss_stop=1e-4, weight_stop=1e-4, max_iter=100 ):
    """
    Fit a logistic regression classifier to data.
    
    # Arguments
        X: an array of sample data, where rows are samples
           and columns are features. caller is responsible
           for prepending x0=1 terms if required.
        y: vector of binary class labels for the samples,
           must be same length as number of rows in X
        w0: starting value of the weights, if omitted
           then all zeros are used
        lr: learning rate, ie fraction of gradients by
           which to update weights at each iteration
        loss_stop: stop iterating if the loss changes
            by less than this (absolute)
        weight_stop: stop iterating if weights change by less
            than this (L2 norm)
        max_iter: stop iterating after iterating this
            many times
           
    # Returns
        ws: a list of fitted weights at each iteration
        losses: a list of the loss values at each iteration
    """
    assert(len(X.shape)==2)
    assert(X.shape[0]==len(y))
    
    if w0 is None: w0 = np.zeros(X.shape[-1])
    
    return gradient_descent ( w0,
                              loss_func = lambda z: logistic_loss(X, y, z),
                              grad_func = lambda z: logistic_grad(X, y, z),
                              lr = lr,
                              loss_stop=loss_stop, z_stop=weight_stop, max_iter=max_iter )


#### plotting utilities

def plot_ridge_regression_1d ( axes, X, y, weights, limits, l2s=[0] ):
    """
    Perform least-squares fits to the provided (X, y) data
    using the specified levels of L2 regularisation, and plot
    the results.
    
    # Arguments
        axes: a Matplotlib Axes object into which to plot
        X: an array of sample data, where rows are samples
            and the single column is the input feature.
        y: vector of output values corresponding to the
            rows of X
        weights: a weight vector of length 2, specifying
            the true generating model, with a bias term
            at index 0.
        limits: a tuple (low, high) specifying the value
            range of the feature dimension x1
        l2s: a list (or vector/array) of numeric values
            specifying amounts of L2 regularisation to use.
    """
    assert(len(X.shape)==2)
    assert(X.shape[1]==1)
    assert(X.shape[0]==len(y))
    
    # plot the data
    axes.scatter(X[:,0], y, marker='x', color='grey')
        
    # plot the true relationship
    y0 = weights[0] + limits[0] * weights[1]
    y1 = weights[0] + limits[1] * weights[1]
    
    axes.plot(limits, (y0, y1), linestyle='dashed', color='red', label='Ground Truth')
    
    # fit for specified regs and plot the results
    X1 = utils.add_x0(X)
    
    cmap = plt.cm.get_cmap('jet')
    for l2 in l2s:
        w = ridge_closed(X1, y, l2)

        y0 = w[0] + limits[0] * w[1]
        y1 = w[0] + limits[1] * w[1]
    
        axes.plot(limits, (y0, y1), linestyle='solid', color=cmap(l2/np.max(l2s)), label='$\lambda=%.f$' % l2)
    
    axes.set_xlim(limits[0], limits[1])
    axes.set_ylim(limits[0], limits[1])
    axes.set_xlabel('$x_1$')
    axes.set_ylabel('$y$')
    
    axes.legend(loc='upper left')
    
    axes.set_title('Ridge Regression')


def plot_poly_fit_1d ( axes, X, y, weights, limits, degrees, l2=0 ):
    """
    Fit polynomials of different degrees to the supplied
    data, and plot the results.
    
    # Arguments
        axes: a Matplotlib Axes object into which to plot
        X: an array of sample data, where rows are samples
            and the single column is the input feature.
        y: vector of output values corresponding to the inputs,
           must be same length as number of rows in X
        weights: the true polynomial coefficients from which
           the data was generated
        limits: a tuple (low, high) specifying the value
            range of the feature dimension x1
        degrees: a list of integer values specifying degrees
            of polynomial to fit
        l2: the amount of l2 regularisation to apply
    
    # Returns
        None
    """
    assert(len(X.shape)==2)
    assert(X.shape[1]==1)
    assert(X.shape[0]==len(y))
    
    axes.scatter(X, y, color='grey', marker='x')
    
    print(f'true weights: {weights}')
    ground_x, ground_y = utils.grid_sample(lambda x: utils.affine(monomial_projection_1d(x, len(weights)-1), weights),
                                           1,
                                           num_divisions=50, limits=limits)
    
    axes.plot(ground_x, ground_y, color='red', linestyle='dashed', label='Ground Truth')

    cmap = plt.cm.get_cmap('jet')
    n = 0
    for deg in degrees:
        w = fit_poly_1d(X, y, deg, l2)
        
        if w is None:
            print('Polynomial fitting not implemented')
            break
            
        print(f'fit {deg} weights: {w}')
        fit_x, fit_y = utils.grid_sample(lambda x: utils.affine(monomial_projection_1d(x, len(w)-1), w),
                                         1,
                                         num_divisions=50, limits=limits)
        axes.plot(fit_x, fit_y, linestyle='solid', color=cmap(n/len(degrees)), label=f'Degree {deg} Fit')
        n += 1
    
    axes.set_xlim(limits[0], limits[1])
    axes.set_xlabel('$x_1$')
    axes.set_ylabel('$y$')
    
    axes.legend(loc='upper right')
    
    axes.set_title('Polynomial Fitting')


def plot_logistic_regression_2d ( axs, X, y, weights, limits ):
    """
    Fit a 2D logistic regression classifier and plot the results.
    Note that there are two separate plots produced here.
    The first (in axs[0]) is an optimisation history, showing how the
    loss decreases via gradient descent. The second (in axs[1]) is
    the regression itself, showing data points and fit results.
    
    # Arguments
        axs: an array of 2 Matplotlib Axes objects into which
           to plot.
        X: an array of sample data, where rows are samples
           and columns are features, including x0=1 terms.
        y: vector of binary class labels for the samples,
           must be same length as number of rows in X
        weights: weights defining the true decision boundary
           with which the data was generated
        limits: a tuple (low, high) specifying the value
            range of both feature dimensions
    
    # Returns
        None
    """
    assert(len(X.shape)==2)
    assert(X.shape[1]==3)
    assert(X.shape[0]==len(y))
    assert(len(weights)==3)
    
    ww, ll = logistic_regression(X, y)
    if ww is None:
        utils.plot_unimplemented(axs[0], title='Logistic Regression Gradient Descent')
        utils.plot_unimplemented(axs[1], title='Logistic Regression Results')
        return
    
    print('Number of iterations: %i' % len(ll))
    axs[0].plot(ll)
    axs[0].set_title('Logistic Regression Gradient Descent')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Logistic Loss')
    
    Xm, ym = utils.grid_sample(lambda x: 1/(1 + np.exp(-utils.affine(x, ww[-1]))), 2, 100, limits)
    axs[1].imshow(ym.T, cmap='coolwarm', origin='lower', extent=(limits[0], limits[1], limits[0], limits[1]), alpha=0.5)
    axs[1].contour(ym.T, levels=[.5], origin='lower', extent=(limits[0], limits[1], limits[0], limits[1]))
    
    y0 = -(weights[0] + limits[0] * weights[1]) / weights[2]
    y1 = -(weights[0] + limits[1] * weights[1]) / weights[2]

    axs[1].plot(limits, (y0, y1), linestyle='dashed', color='red', marker='')

    axs[1].plot(X[y==0,1], X[y==0,2], linestyle='', color='orange', marker='v', label='Class 0')
    axs[1].plot(X[y==1,1], X[y==1,2], linestyle='', color='darkorchid', marker='o', label='Class 1')
    
    axs[1].set_xlabel('$x_1$')
    axs[1].set_ylabel('$x_2$')
    
    axs[1].legend(loc='upper left', framealpha=1)
    
    axs[1].set_title('Logistic Regression Results')


#### TEST DRIVER

def process_args():
    ap = argparse.ArgumentParser(description='week 2 coursework script for COMP0088')
    ap.add_argument('-s', '--seed', help='seed random number generator', type=int, default=None)
    ap.add_argument('-n', '--num_samples', help='number of samples to generate and fit', type=int, default=50)
    ap.add_argument('file', help='name of output file to produce', nargs='?', default='week_2.pdf')
    return ap.parse_args()

if __name__ == '__main__':
    args = process_args()
    rng = numpy.random.default_rng(args.seed)
    
    LIMITS = (-5, 5)
    WEIGHTS = np.array([0.5, -0.4, 0.6])
    fig = plt.figure(figsize=(10, 10))
    axs = fig.subplots(nrows=2, ncols=2)
    
    print('Q1: testing unregularised least squares')
    X, y = generate_noisy_linear(args.num_samples, WEIGHTS, 0.5, LIMITS, rng)
    if X is None:
        print('(week 1) linear generation not implemented')
        utils.plot_unimplemented(axs[0,0], title='Ridge Regression')
    else:
        w = ridge_closed(utils.add_x0(X), y)

        print('true weights: %.2f, %.2f, %.2f' % (WEIGHTS[0], WEIGHTS[1], WEIGHTS[2]))
    
        if w is None:
            print('regression not implemented')
            utils.plot_unimplemented(axs[0,0], title='Ridge Regression')
        else:
            print('regressed weights: %.2f, %.2f, %.2f' % (w[0], w[1], w[2]))
            print('squared error: %.2g' % np.dot(WEIGHTS-w, WEIGHTS-w))
    
            print('plotting regularised least squares')
            X, y = generate_noisy_linear(args.num_samples, WEIGHTS[1:], 3, LIMITS, rng)
            plot_ridge_regression_1d ( axs[0, 0], X, y, WEIGHTS[1:], LIMITS, np.arange(5) * 25 )
    
    print('\nQ2: plotting 1D polynomial fits')
    X, y = generate_noisy_poly_1d ( args.num_samples, WEIGHTS, 3, LIMITS, rng )
    if X is None:
        print('poly generation not implemented')
        utils.plot_unimplemented(axs[0,1], title='Polynomial Fitting')
    else:
        plot_poly_fit_1d ( axs[0, 1], X, y, WEIGHTS, LIMITS, [1, 2, 3, 4], 0 )
        
    print('\nQ3: testing gradient descent')
    xx, ll = gradient_descent ( 10,
                                lambda x: x * x - 2 * x + 1,
                                lambda x: 2 * x - 2,
                                lr=0.1,
                                loss_stop=1e-6,
                                z_stop=1e-6 )
    if xx is None:
        print('gradient descent not implemented')
    else:
        print('final estimate %.3g after %i iterations (loss %.2g)' % (xx[-1], len(ll), ll[-1]))
    
    print('\nQ4: testing logistic regression with 2 feature dimensions')
    X, y = generate_linearly_separable(args.num_samples, WEIGHTS, LIMITS, rng)
    if X is None:
        print('(week 1) linearly-separable generation not implemented')
        utils.plot_unimplemented(axs[1,0], title='Logistic Regression Gradient Descent')
        utils.plot_unimplemented(axs[1,1], title='Logistic Regression Results')
    else:
        X0 = utils.add_x0(X)
    
        plot_logistic_regression_2d(axs[1, :], X0, y, WEIGHTS, LIMITS)

    fig.tight_layout(pad=1)
    fig.savefig(args.file)
    plt.close(fig)
