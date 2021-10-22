#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
COMP0088 lab exercises for week 4.

Add your code as specified below.

A simple test driver is included in this script. Call it at the command line like this:

  $ python week_4.py

A 6-panel figure, `week_4.pdf`, will be generated so you can check it's doing what you
want. You should not need to edit the driver code, though you can if you wish.
"""

import sys, os, os.path
import argparse
import pprint
from time import perf_counter

import numpy as np
import numpy.random
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.svm import SVC

import utils


#### ADD YOUR CODE BELOW

# -- Question 1 --

def generate_margined_binary_data ( num_samples, count, limits, rng ):
    """
    Draw random samples from a linearly-separable binary model
    with some non-negligible margin between classes. (The exact
    form of the model is up to you.)
    
    # Arguments
        num_samples: number of samples to generate
            (ie, the number of rows in the returned X
            and the length of the returned y)
        count: the number of feature dimensions
        limits: a tuple (low, high) specifying the value
            range of all the features x_i
        rng: an instance of numpy.random.Generator
            from which to draw random numbers

    # Returns
        X: a matrix of sample vectors, where
            the samples are the rows and the
            features are the columns
            ie, its size should be:
              num_samples x count
        y: a vector of num_samples binary labels
    """
    # TODO: implement this
    return None, None


def geometric_margin ( X, y, weights, bias ):
    """
    Calculate the geometric margin for a given
    dataset and linear decision boundary. May be
    negative if any of the samples are
    misclassified.

    # Arguments
        X: an array of sample data, where rows are samples
           and columns are features.
        y: vector of ground truth labels for the samples,
           must be same length as number of rows in X
        weights: a vector of weights defining the direction
           of the decision boundary, must be the same
           length as the number of features
        bias: scalar intercept value specifying the position
           of the boundary
    
    # Returns:
        g: the geometric margin -- ie, the minimum distance
           of any of the samples on the correct side of the
           boundary (or the negative greatest distance on the
           wrong side)
    """
    assert(X.shape[0] == len(y))
    assert(X.shape[1] == len(weights))
    
    # TODO: implement this
    return None
    

# -- Question 2 --

def perceptron_train ( X, y, alpha=1, max_epochs=50, include_bias=True ):
    """
    Learn a linear decision boundary using the
    perceptron algorithm.
    
    # Arguments
        X: an array of sample data, where rows are samples
            and columns are features.
        y: vector of ground truth labels for the samples,
            must be same length as number of rows in X
        alpha: learning rate, ie how much to adjust the
            boundary for each misclassified sample
        max_epochs: maximum number of passes over the
            training set before admitting defeat
        include_bias: whether to automatically add a
            a constant bias feature x0
    
    # Returns:
        weights: vector of feature weights defining the
            decision boundary, either same length as number of
            columns in X or 1 greater if include_bias is True.
            (note that a weights vector will be returned even if
            the algorithm fails to converge)
    """
    assert(X.shape[0] == len(y))
    
    # TODO: implement this
    return None


def perceptron_predict ( test_X, weights ):
    """
    Predict binary labels for a dataset using a specified
    decision boundary. (This is intended for us with a boundary
    learned with the perceptron algorithm, but any suitable
    weights vector can be used.)
    
    # Arguments
        test_X: an array of sample data, where rows are samples
            and columns are features.
        weights: vector of feature weights defining the
            decision boundary, either same length as number of
            columns in X or 1 greater -- in the latter case it
            is assumed to contain a bias term, and test_X will
            have a constant term x0=1 prepended
    
    # Returns
        pred_y: a vector of predicted binary labels
            corresponding to the samples in test_X
        
    """
    assert(test_X.shape[1] in (len(weights),len(weights)-1))
    
    # TODO: implement this
    return None
    

# -- Question 3 --

def generate_binary_nonlinear_2d ( num_samples, limits, rng ):
    """
    Draw random samples from a binary model that is *not*
    linearly separable in its 2D feature space. (The exact
    form of the model is up to you.)
    
    # Arguments
        num_samples: number of samples to generate
            (ie, the number of rows in the returned X
            and the length of the returned y)
        limits: a tuple (low, high) specifying the value
            range of all the features x_i
        rng: an instance of numpy.random.Generator
            from which to draw random numbers

    # Returns
        X: a matrix of sample vectors, where
            the samples are the rows and the
            features are the columns
            ie, its size should be:
              num_samples x count
        y: a vector of num_samples binary labels
    """
    # TODO: implement this
    return None, None


# -- Question 4 --
    
def custom_kernel ( X1, X2, gamma=0.5 ):
    """
    Custom kernel function for use with a support
    vector classifier.
    
    # Arguments
        X1: first array of sample data for comparison,
            with samples as rows and features as
            columns (size N1 x M)
        X2: second array of sample data for comparison,
            with samples as rows and features as
            columns (size N2 x M; may be the same
            as X1)
        gamma: a scaling hyperparameter for the similarity
            function
    
    # Returns
        K: the Gram matrix for the kernel, giving the
           kernel space inner products for each pairing of
           a vector in X1 with one in X2 (size N1 x N2)
    """
    assert(X1.shape[1] == X2.shape[1])
    
    # TODO: implement this
    return None


#### TEST DRIVER

def process_args():
    ap = argparse.ArgumentParser(description='week 4 coursework script for COMP0088')
    ap.add_argument('-s', '--seed', help='seed random number generator', type=int, default=None)
    ap.add_argument('-n', '--num_samples', help='number of samples to use', type=int, default=50)
    ap.add_argument('-c', '--cost', help='cost hyperparam for SVMs', type=float, default=1.0)
    ap.add_argument('-g', '--gamma', help='gamma hyperparam for SVM kernels', type=float, default=0.5)
    ap.add_argument('-r', '--resolution', help='grid sampling resolution for classification plots', type=int, default=20)
    ap.add_argument('file', help='name of output file to produce', nargs='?', default='week_4.pdf')
    return ap.parse_args()

def plot_svm_map ( axes, svm, X, y, resolution, title ):
    """
    Utility to plot the classification map of an SVM
    highlighting the support vectors.
    """
    utils.plot_classification_map(axes, lambda z: svm.predict(z), X, y,
                                  resolution=resolution, title=title, legend_loc=None)
    axes.scatter(X[svm.support_,0], X[svm.support_,1],
                 facecolors='none', edgecolors='k', s=140, label='SV')
    axes.legend(loc='upper left')


if __name__ == '__main__':
    args = process_args()
    rng = numpy.random.default_rng(args.seed)
    
    LIMITS=(-5, 5)
    
    fig = plt.figure(figsize=(12, 8))
    axs = fig.subplots(nrows=2, ncols=3)

    print('Q1: generating linearly separable data')    
    X_lin, y_lin = generate_margined_binary_data ( args.num_samples, 2, LIMITS, rng )
    if X_lin is None:
        print('not implemented')
        utils.plot_unimplemented(axs[0,0], title='Linear Data, Linear SVM')
        utils.plot_unimplemented(axs[0,1], title='Linear Data, Perceptron')
        utils.plot_unimplemented(axs[0,2], title='Linear Data, RBF SVM')        
    else:
        print('Q1: fitting linear SVM')
        t0 = perf_counter()
        svm_lin = SVC(kernel='linear', C=args.cost)
        svm_lin.fit(X_lin, y_lin)
        print('time taken: %.2f seconds' % (perf_counter() - t0))

        print('Q1: plotting fit')
        t0 = perf_counter()
        plot_svm_map ( axs[0,0], svm_lin, X_lin, y_lin, args.resolution, title='Linear Data, Linear SVM' )
        print('time taken: %.2f seconds' % (perf_counter() - t0))
        
        print('Q1: calculating geometric margin')
        marg_svm = geometric_margin(X_lin, y_lin, svm_lin.coef_[0,:], svm_lin.intercept_[0])
        
        if marg_svm is None:
            print('not implemented')
        else:
            print(f'Linear SVM geometric margin: {marg_svm:.3f}')

        print('Q1: fitting RBF SVM')
        t0 = perf_counter()
        svm_rbf = SVC(kernel='rbf', C=args.cost, gamma=args.gamma)
        svm_rbf.fit(X_lin, y_lin)
        print('time taken: %.2f seconds' % (perf_counter() - t0))

        print('Q1: plotting fit')
        t0 = perf_counter()
        plot_svm_map ( axs[0,2], svm_rbf, X_lin, y_lin, args.resolution, title='Linear Data, RBF SVM' )
        print('time taken: %.2f seconds' % (perf_counter() - t0))

        print('Q2: fitting perceptron')

        t0 = perf_counter()
        pw = perceptron_train(X_lin, y_lin)
        if pw is None:
            print('perceptron not implemented')
            utils.plot_unimplemented(axs[0,1], title='Linear Data, Perceptron')
        else:
            print('time taken: %.2f seconds' % (perf_counter() - t0))
            print('Q2: plotting fit')
            utils.plot_classification_map(axs[0,1], lambda z: perceptron_predict(z, pw), X_lin, y_lin, resolution=args.resolution, title=f'Linear Data, Perceptron')

            print('Q2: calculating geometric margin')
            marg_ptron = geometric_margin(X_lin, y_lin, pw[1:], pw[0])
        
            if marg_svm is None:
                print('not implemented')
            else:
                print(f'Perceptron geometric margin: {marg_ptron:.3f}')
                print(f'SVM margin greater by: {marg_svm - marg_ptron:.3f}')

    print('Q3: generating non-linear data')    
    X_non, y_non = generate_binary_nonlinear_2d ( args.num_samples, LIMITS, rng )
    
    if X_non is None:
        print('not implemented')
        utils.plot_unimplemented(axs[1,0], title='Non-Linear Data, Perceptron')
        utils.plot_unimplemented(axs[1,1], title='Non-Linear Data, RBF SVM')
        utils.plot_unimplemented(axs[1,2], title='Non-Linear Data, Custom SVM')
    else:
        print('Q3: fitting perceptron')
        t0 = perf_counter()
        pw = perceptron_train(X_non, y_non)
        if pw is None:
            print('perceptron not implemented')
            utils.plot_unimplemented(axs[1,0], title='Non-Linear Data, Perceptron')
        else:
            print('time taken: %.2f seconds' % (perf_counter() - t0))
            print('Q3: plotting fit')
            utils.plot_classification_map(axs[1,0], lambda z: perceptron_predict(z, pw), X_non, y_non, resolution=args.resolution, title=f'Non-Linear Data, Perceptron')
        
        print('Q3: fitting RBF SVM')
        t0 = perf_counter()
        svm_rbf = SVC(kernel='rbf', C=args.cost, gamma=args.gamma)
        svm_rbf.fit(X_non, y_non)
        print('time taken: %.2f seconds' % (perf_counter() - t0))

        print('Q3: plotting fit')
        t0 = perf_counter()
        plot_svm_map ( axs[1,1], svm_rbf, X_non, y_non, args.resolution, title='Non-Linear Data, RBF SVM')
        print('time taken: %.2f seconds' % (perf_counter() - t0))
        
        # check custom kernel at least superficially works
        # before trying to train an SVM with it
        zz = np.zeros((1,1))
        if custom_kernel(zz,zz) is None:
            print('Q4: custom kernel not implemented')
            utils.plot_unimplemented(axs[1,2], title='Non-Linear Data, Custom SVM')
        else:
            print(f'Q4: fitting SVM with custom kernel')
            
            def kernel ( x, y ):
                return custom_kernel(x, y, args.gamma)
            
            t0 = perf_counter()
            svm_cust = SVC(kernel=custom_kernel, C=args.cost)
            svm_cust.fit(X_non, y_non)
            print('time taken: %.2f seconds' % (perf_counter() - t0))

            print('Q4: plotting fit')
            t0 = perf_counter()
            plot_svm_map ( axs[1,2], svm_cust, X_non, y_non, args.resolution, title='Non-Linear Data, Custom SVM')
            print('time taken: %.2f seconds' % (perf_counter() - t0))
    
    fig.tight_layout(pad=1)
    fig.savefig(args.file)
    plt.close(fig)
