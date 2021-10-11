#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
COMP0088 lab exercises for week 1.

This first introductory set of exercises is largely intended
as a warm up and practice session. It is an opportunity to check
that you have a functioning Python 3 system with the requisite libraries, to get
a feel for some basic data manipulation and plotting, and to ensure that
everything makes sense and runs smoothly.

Add your code as specified below. You shouldn't need to load further external
code that isn't already explicitly imported.

A simple test driver is included in this script. Call it at the command line like this:

  $ python week_1.py

A 4-panel figure, `week_1.pdf`, will be generated so you can check it's doing what you
want. You should not need to edit the driver code, though you can if you wish.
"""

import sys, os, os.path
import argparse

import numpy as np
import numpy.random
import matplotlib.pyplot as plt

import utils


#### ADD YOUR CODE BELOW

# -- Question 1 --

def generate_noisy_linear(num_samples, weights, sigma, limits, rng):
    """
    Draw samples from a linear model with additive Gaussian noise.
    
    # Arguments
        num_samples: number of samples to generate
            (ie, the number of rows in the returned X
            and the length of the returned y)
        weights: vector defining the model
            (including a bias term at index 0)
        sigma: standard deviation of the additive noise
        limits: a tuple (low, high) specifying the value
            range of all the input features x_i
        rng: an instance of numpy.random.Generator
            from which to draw random numbers
    
    # Returns
        X: a matrix of sample inputs, where
            the samples are the rows and the
            features are the columns
            ie, its size should be:
              num_samples x (len(weights) - 1)
        y: a vector of num_samples output values
    """    
    return utils.random_sample(lambda x: utils.affine(x, weights),
                               len(weights) - 1,
                               num_samples, limits, rng, sigma)


def plot_noisy_linear_1d(axes, num_samples, weights, sigma, limits, rng):
    """
    Generate and plot points from a noisy single-feature linear model,
    along with a line showing the true (noiseless) relationship.
    
    # Arguments
        axes: a Matplotlib Axes object into which to plot
        num_samples: number of samples to generate
            (ie, the number of rows in the returned X
            and the length of the returned y)
        weights: vector defining the model
            (including a bias term at index 0)
        sigma: standard deviation of the additive noise
        limits: a tuple (low, high) specifying the value
            range of all the input features x_i
        rng: an instance of numpy.random.Generator
            from which to draw random numbers
    
    # Returns
        None
    """
    assert(len(weights)==2)
    X, y = generate_noisy_linear(num_samples, weights, sigma, limits, rng)
    
    axes.plot(X, y, color='red', marker='o', linestyle='')
    
    y0 = weights[0] + limits[0] * weights[1]
    y1 = weights[0] + limits[1] * weights[1]
    axes.plot(limits, (y0, y1), linestyle='dashed', color='green', marker='')

    axes.set_title('Noisy 1D Linear Model')
    axes.set_xlim(limits[0], limits[1])
    axes.set_ylim(limits[0], limits[1])
    axes.set_xlabel('$x$')
    axes.set_ylabel('$y$')


def plot_noisy_linear_2d(axes, resolution, weights, sigma, limits, rng):
    """
    Produce a plot illustrating a noisy two-feature linear model.
    
    # Arguments
        axes: a Matplotlib Axes object into which to plot
        resolution: how densely should the model be sampled?
        weights: vector defining the model
            (including a bias term at index 0)
        sigma: standard deviation of the additive noise
        limits: a tuple (low, high) specifying the value
            range of all the input features x_i
        rng: an instance of numpy.random.Generator
            from which to draw random numbers
    
    # Returns
        None
    """
    assert(len(weights)==3)
    
    X, y = utils.grid_sample(lambda x: utils.affine(x, weights), 2, resolution, limits, rng, sigma)
                       
    axes.imshow(y.T, cmap='GnBu', origin='lower', extent=(limits[0], limits[1], limits[0], limits[1]) )
    
    levels = np.linspace(np.min(y), np.max(y), 10)
    axes.contour(y.T, levels, colors='white', origin='lower', extent=(limits[0], limits[1], limits[0], limits[1]) )

    axes.set_xlabel('$x_1$')
    axes.set_ylabel('$x_2$')
    
    axes.set_title('Noisy 2D Linear Model')


# -- Question 2 --

def generate_linearly_separable(num_samples, weights, limits, rng):
    """
    Draw samples from a binary model with a given linear
    decision boundary.

    # Arguments
        num_samples: number of samples to generate
            (ie, the number of rows in the returned X
            and the length of the returned y)
        weights: vector defining the decision boundary
            (including a bias term at index 0)
        limits: a tuple (low, high) specifying the value
            range of all the input features x_i
        rng: an instance of numpy.random.Generator
            from which to draw random numbers

    # Returns
        X: a matrix of sample vectors, where
            the samples are the rows and the
            features are the columns
            ie, its size should be:
              num_samples x (len(weights) - 1)
        y: a vector of num_samples binary labels
    """
    return utils.random_sample(lambda x: hyperplane_label(x, weights),
                               count = len(weights) - 1,
                               num_samples = num_samples,
                               limits = limits,
                               rng = rng)


def plot_linearly_separable_2d(axes, num_samples, weights, limits, rng):
    """
    Plot a linearly separable binary data set in a 2d feature space.

    # Arguments
        axes: a Matplotlib Axes object into which to plot
        num_samples: number of samples to generate
            (ie, the number of rows in the returned X
            and the length of the returned y)
        weights: vector defining the decision boundary
            (including a bias term at index 0)
        limits: a tuple (low, high) specifying the value
            range of all the input features x_i
        rng: an instance of numpy.random.Generator
            from which to draw random numbers

    # Returns
        None
    """
    assert(len(weights)==3)
    X, y = generate_linearly_separable(num_samples, weights, limits, rng)
    
    # plot the two subsets with different markers & colours
    axes.plot(X[y < 0.5, 0], X[y < 0.5, 1], color='red', marker='o', linestyle='', label='Class 0')
    axes.plot(X[y >= 0.5, 0], X[y >= 0.5, 1], color='blue', marker='v', linestyle='', label='Class 1')

    # to draw the boundary line we need to calculate the endpoints
    # NB: this will fail if either of the coordinate weights is 0.
    # special case it later
    y0 = -(weights[0] + limits[0] * weights[1]) / weights[2]
    y1 = -(weights[0] + limits[1] * weights[1]) / weights[2]

    axes.plot(limits, (y0, y1), linestyle='dashed', color='green', marker='')
    
    mid_x = np.sum(limits)/2
    mid_y = (y0 + y1)/2
    
    axes.arrow(mid_x, mid_y, weights[1], weights[2], color='darkorchid', width=0.06, head_width=0.3, overhang=0.3)

    axes.legend(loc='upper left')
    
    axes.set_title('Linearly Separable Binary Data')
    axes.set_xlim(limits[0], limits[1])
    axes.set_ylim(limits[0], limits[1])
    axes.set_xlabel('$x_1$')
    axes.set_ylabel('$x_2$')


# -- Question 3 --

def random_search(function, count, num_samples, limits, rng):
    """
    Randomly sample from a function of `count` features and return
    the best feature vector found.
    
    # Arguments
        function: a function taking a single input array of
            shape (..., count), where the last dimension
            indexes the features
        count: the number of features expected by the function
        num_samples: the number of samples to generate & search
        limits: a tuple (low, high) specifying the value
            range of all the input features x_i
        rng: an instance of numpy.random.Generator
            from which to draw random numbers
    
    # Returns
        x: a vector of length count, containing the found features
    """
    X, y = utils.random_sample(function, count, num_samples, limits, rng)
    loc = np.argmin(y)
    
    return X[loc, :]

def grid_search(function, count, num_divisions, limits):
    """
    Perform a grid search for a function of `count` features and
    return the best feature vector found.
    
    # Arguments
        function: a function taking a single input array of
            shape (..., count), where the last dimension
            indexes the features
        count: the number of features expected by the function
        num_divisions: the number of samples along each feature
            dimension (including endpoints)
        limits: a tuple (low, high) specifying the value
            range of all the input features x_i
    
    # Returns
        x: a vector of length count, containing the found features
    """
    X, y = utils.grid_sample(function, count, num_divisions, limits)
    loc = np.unravel_index(np.argmin(y), y.shape)
    
    return X[loc]


def hyperplane_label(X, boundary):
    y = utils.affine(X, boundary)
    return (y > 0).astype(np.float64)


def plot_searches_2d(axes, function, limits, resolution,
                     num_divisions, num_samples, rng, true_min=None):
    """
    Plot a 2D function aling with minimum values found by
    grid and random searching.

    # Arguments
        axes: a Matplotlib Axes object into which to plot
        function: a function taking a single input array of
            shape (..., 2), where the last dimension
            indexes the features
        limits: a tuple (low, high) specifying the value
            range of both input features x1 and x2
        resolution: number of samples along each side
            (including endpoints) for an image representation
            of the function
        num_divisions: the number of samples along each side
            (including endpoints) for a grid search for
            the function minimum
        num_samples: number of samples to draw for a random
            search for the function minimum
        rng: an instance of numpy.random.Generator
            from which to draw random numbers
        true_min: an optional (x1, x2) tuple specifying
            the location of the actual function minimum
            
    # Returns
        None
    """
    X, y = utils.grid_sample(function, 2, resolution, limits)
    axes.imshow(y.T, cmap='GnBu', origin='lower', extent=(limits[0], limits[1], limits[0], limits[1]) )
    
    levels = np.linspace(np.min(y), np.max(y), 10)
    axes.contour(y.T, levels, colors='white', origin='lower', extent=(limits[0], limits[1], limits[0], limits[1]) )
    
    r1, r2 = random_search(function, 2, num_samples, limits, rng)
    g1, g2 = grid_search(function, 2, num_divisions, limits)
    
    if true_min is not None:
        axes.plot(true_min[0], true_min[1], color='green', marker='x', linestyle='', label='True')

    axes.plot(r1, r2, color='red', marker='o', linestyle='', label='Random')
    axes.plot(g1, g2, color='blue', marker='v', linestyle='', label='Grid')
        
    axes.legend()
    
    axes.set_xlabel('$x_1$')
    axes.set_ylabel('$x_2$')
    
    axes.set_title('Sampling Search')


#### TEST DRIVER

def process_args():
    ap = argparse.ArgumentParser(description='week 1 coursework script for COMP0088')
    ap.add_argument('-s', '--seed', help='seed random number generator', type=int, default=None)
    ap.add_argument('file', help='name of output file to produce', nargs='?', default='week_1.pdf')
    return ap.parse_args()


def test_func(X):
    """
    Simple example function of 2 variables for
    testing grid & random optimisation.
    """
    return (X[..., 0]-1)**2 + X[...,1]**2 + 2 * np.abs((X[...,0]-1) * X[...,1])

WEIGHTS = np.array([0.5, -0.4, 0.6])
LIMITS = (-5, 5)

if __name__ == '__main__':
    args = process_args()
    rng = numpy.random.default_rng(args.seed)

    fig = plt.figure(figsize=(8, 8))
    axs = fig.subplots(nrows=2, ncols=2)
    
    print('Q1: noisy continuous data')
    print('plotting 1D data')
    plot_noisy_linear_1d(axs[0, 0], 50, WEIGHTS[1:], 0.5, LIMITS, rng)
    print('plotting 2D data')
    plot_noisy_linear_2d(axs[0, 1], 100, WEIGHTS, 0.2, LIMITS, rng)
    
    print('\nQ2: binary separable data')
    print('plotting 2D labelled data')
    plot_linearly_separable_2d(axs[1, 0], num_samples=100, weights=WEIGHTS, limits=LIMITS, rng=rng)
    
    print('\nQ3: searching for a minimiser')
    print('plotting searches')
    plot_searches_2d(axs[1, 1], test_func, limits=LIMITS, resolution=100, num_divisions=10, num_samples=100, rng=rng, true_min=(1,0))

    fig.tight_layout(pad=1)
    fig.savefig(args.file)
    plt.close(fig)
