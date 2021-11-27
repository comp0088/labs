#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
COMP0088 lab exercises for week 8.

Add your code as specified below.

A simple test driver is included in this script. Call it at the command line like this:

  $ python week_4.py

A 6-panel figure, `week_8.pdf`, will be generated so you can check it's doing what you
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

from scipy.stats import multivariate_normal as mvn, norm

try:
    import hmmlearn.hmm as hmm
except ModuleNotFoundError:
    hmm = None

import utils


#### ADD YOUR CODE BELOW

# -- Question 1 --

def generate_gaussian_mix ( num_samples, means, covs,
                            class_probs, rng ):
    """
    Draw labelled samples from a mixture of multivariate
    gaussians.
    
    # Arguments
        num_samples: number of samples to generate
            (ie, the number of rows in the returned X
            and the length of the returned y)
        means: a list of vectors specifying mean of each gaussian
            (all the same length == the number of features)
        covs: a list of covariance matrices
            (same length as means, with each matrix being
            num features x num features, symmetric and
            positive semidefinite)
        class_probs: a vector of class probabilities,
            (same length as means, all non-negative and
            summing to 1)
        rng: an instance of numpy.random.Generator
            from which to draw random numbers
    
    # Returns
        X: a matrix of sample inputs, where
            the samples are the rows and the
            columns are features, ie size is:
              num_samples x num_features
        y: a vector of num_samples labels matching
            the samples to the gaussian from which
            they were drawn
    """
    assert(len(means)==len(covs)==len(class_probs))
    
    # TODO: implement this
    return None, None


# -- Question 2 --

def gaussian_mix_loglik ( X, means, covs, class_probs ):
    """
    Estimate the log likelihood of the given mixture model.
    
    # Arguments
        X: a matrix of sample inputs, where
            the samples are the rows and the
            columns are features, ie size is:
              num_samples x num_features
        means: a list of vectors specifying mean of each gaussian
            (all the same length == the number of features)
        covs: a list of covariance matrices
            (same length as means, with each matrix being
            num features x num features, symmetric and
            positive semidefinite)
        class_probs: a vector of class probabilities,
            (same length as means, all non-negative and
            summing to 1)
    
    # Returns
        loglik: the (scalar) log likelihood of the model
    """

    # TODO: implement this
    return None

def gaussian_mix_E_step ( X, means, covs, class_probs ):
    """
    Given a candidate set of gaussian mix parameters,
    estimate the responsiblilites of each component for
    each data sample.
    
    # Arguments
        X: a matrix of sample inputs, where
            the samples are the rows and the
            columns are features, ie size is:
              num_samples x num_features
        means: a list of vectors specifying mean of each gaussian
            (all the same length == the number of features)
        covs: a list of covariance matrices
            (same length as means, with each matrix being
            num features x num features, symmetric and
            positive semidefinite)
        class_probs: a vector of class probabilities,
            (same length as means, all non-negative and
            summing to 1)
    
    # Returns
        resps: a matrix of weights attributing
            samples to source gaussians, of size
              num_samples x num_gaussians
    """
    assert(len(means)==len(covs)==len(class_probs))

    # TODO: implement this
    return None


def gaussian_mix_M_step ( X, resps ):
    """
    Given a candidate set of responsibilities,
    estimate new gaussian mixture model parameters.
    
    # Arguments
        X: a matrix of sample inputs, where
            the samples are the rows and the
            columns are features, ie size is:
              num_samples x num_features
        resps: a matrix of weights attributing
            samples to source gaussians, of size
              num_samples x num_gaussians

    # Returns
        means: a list of vectors specifying mean of each gaussian
            (all the same length == the number of features)
        covs: a list of covariance matrices
            (same length as means, with each matrix being
            num features x num features, symmetric and
            positive semidefinite)
        class_probs: a vector of class probabilities,
            (same length as means, all non-negative and
            summing to 1)
    """

    # TODO: implement this
    return None, None, None


def fit_gaussian_mix ( X, num_gaussians, rng, max_iter=10,
                       loglik_stop=1e-1 ):
    """
    Fit a gaussian mixture model to some data.
    
    # Arguments
        X: a matrix of sample inputs, where
            the samples are the rows and the
            columns are features, ie size is:
              num_samples x num_features
        num_gaussians: the number of components
            to fit
        rng: an instance of numpy.random.Generator
            from which to draw random numbers
        max_iter: the maximum number of iterations
            to perform
        loglik_stop: stop iterating once the improvement
            in log likelihood drops below this
    
    # Returns
        resps: a matrix of weights attributing
            samples to source gaussians, of size
              num_samples x num_gaussians
        means: a list of num_gaussians vectors specifying means
        covs: a list of num_gaussians covariance matrices
        class_probs: a vector of num_gaussian class probabilities
        logliks: a list of the model log likelihood values after
            each fitting iteration
    """
    
    # TODO: implement this
    return None, None, None, None, None


# -- Question 3 --

def generate_hmm_sequence ( num_samples,
                            initial_probs, transitions,
                            emission_means, emission_sds,
                            rng ):
    """
    Generate a sequence of observations from a hidden Markov
    model with the given parameters. Emissions are univariate
    Gaussians.
    
    # Arguments
        num_samples: number of samples (ie timesteps) to generate
        initial_probs: vector of probabilities of being in each hidden
            state at time step 1; must sum to 1
        transitions: matrix of transition probabilities, from state
            indexed by row to state indexed by column; rows must sum to 1
        emission_means: mean of observations for each hidden state
        emission_sds: standard deviation of observations for each hidden state
        rng: an instance of numpy.random.Generator
            from which to draw random numbers
    
    # Returns
        x: a vector of observations for each time step
        z: a vector of hidden state (indices) for each time step
    """
    # TODO: implement this
    return None, None
        

# -- Question 4 --

def viterbi ( x, initial_probs, transitions, emission_means, emission_sds ):
    """
    Infer the most likely sequence of hidden states based on
    observations and HMM parameters.
    
    # Arguments
        x: a vector of observations for each time steps
        initial_probs: vector of probabilities of being in each hidden
            state at time step 1; must sum to 1
        transitions: matrix of transition probabilities, from state
            indexed by row to state indexed by column; rows must sum to 1
        emission_means: mean of observations for each hidden state
        emission_sds: standard deviation of observations for each hidden state
    
    # Returns
        z: a vector of predicted hidden state (indices) for each time step
    """
    # TODO: implement this
    return None
    
    

#### TEST DRIVER

def process_args():
    ap = argparse.ArgumentParser(description='week 4 coursework script for COMP0088')
    ap.add_argument('-s', '--seed', help='seed random number generator', type=int, default=None)
    ap.add_argument('-n', '--num_samples', help='number of samples to use', type=int, default=100)
    ap.add_argument('-g', '--num_gaussians', help='number of gaussians to generate', type=int, default=2)
    ap.add_argument('-f', '--fit_gaussians', help='number of gaussians to fit', type=int, default=None)
    ap.add_argument('-m', '--max_iter', help='max number of iterations to fit', type=int, default=10)
    ap.add_argument('-k', '--num_hidden', help='number of hidden HMM states', type=int, default=3)
    ap.add_argument('-d', '--dwell', help='bias towards remaining in same state', type=float, default=2)
    ap.add_argument('-N', '--noise', help='noise scaling on HMM observations', type=float, default=0.3)
    
    ap.add_argument('file', help='name of output file to produce', nargs='?', default='week_8.pdf')
    return ap.parse_args()


if __name__ == '__main__':
    args = process_args()
    rng = numpy.random.default_rng(args.seed)
    
    args.fit_gaussians = args.fit_gaussians or args.num_gaussians
    
    LIMITS = (-10, 10)
    DARK = [plt.cm.tab20.colors[2 * ii] for ii in range(10)]
    LIGHT = [plt.cm.tab20.colors[2 * ii + 1] for ii in range(10)]
    
    fig = plt.figure(figsize=(12, 8))
    axs = fig.subplots(nrows=2, ncols=3)
    
    # Questions 1-2: Gaussian mixtures

    print(f'Q1: generating {args.num_samples} samples from mixture of {args.num_gaussians} Gaussians')    
    means = [ rng.random(2) * (LIMITS[1] - LIMITS[0]) + LIMITS[0] for ii in range(args.num_gaussians) ]
    covs = [ rng.random((2,2)) * 2 - 1 for ii in range(args.num_gaussians) ]
    covs = [ (np.eye(2) + cc.T @ cc) for cc in covs ]
    class_probs = rng.random(args.num_gaussians) + np.ones(args.num_gaussians) * 0.5
    class_probs /= np.sum(class_probs)
    
    X, y = generate_gaussian_mix ( args.num_samples, means, covs, class_probs, rng )
    
    if X is None:
        print('-> not implemented')
        utils.plot_unimplemented(axs[0,0], title=f'Source data from {args.num_gaussians} Gaussians')
        utils.plot_unimplemented(axs[0,1], title=f'Fitting History')
        utils.plot_unimplemented(axs[0,2], title=f'Fit to {args.fit_gaussians} Gaussians')
    else:
        print('Plotting true Gaussian mix')
        for ii in range(args.num_gaussians):
            idx = (y == ii)
            axs[0,0].scatter(X[idx,0], X[idx,1], color=LIGHT[ii])
    
        left, right = axs[0,0].get_xlim()
        bottom, top = axs[0,0].get_ylim()    
        xx = np.linspace(left, right, 100)
        yy = np.linspace(bottom, top, 100)
        grid = np.moveaxis(np.stack(np.meshgrid(xx, yy, indexing='xy')), 0, -1)
        extent = (left, right, bottom, top)    
    
        for ii in range(args.num_gaussians):
            pdf = mvn.pdf(grid, mean=means[ii], cov=covs[ii])
            axs[0,0].contour(xx, yy, pdf, origin='lower', extent=extent, alpha=0.5, colors=[DARK[ii]])
            axs[0,0].scatter(means[ii][0], means[ii][1], s=100, color=DARK[ii], marker='x')
    
        axs[0,0].set_title(f'Source data from {args.num_gaussians} Gaussians')
        axs[0,0].set_xlabel('$x_1$')
        axs[0,0].set_ylabel('$x_2$')

        print(f'Q2: attempting to fit data with {args.fit_gaussians} Gaussians')    
        resps, fit_means, fit_covs, fit_class_probs, logliks = fit_gaussian_mix ( X, num_gaussians=args.fit_gaussians, rng=rng, max_iter=args.max_iter )
        
        if resps is None:
            print('-> not implemented')
            utils.plot_unimplemented(axs[0,1], title=f'Fitting History')
            utils.plot_unimplemented(axs[0,2], title=f'Fit to {args.fit_gaussians} Gaussians')
        else:
            axs[0,1].plot(np.arange(len(logliks)) + 1, logliks)
            axs[0,1].set_title(f'Fitting History')
            axs[0,1].set_xlabel('Iteration')
            axs[0,1].set_ylabel('Model Log Likelihood')

            y_hat = np.argmax(resps, axis=1)
            for ii in range(args.fit_gaussians):
                idx = (y_hat == ii)
                axs[0,2].scatter(X[idx,0], X[idx,1], color=LIGHT[ii])

            left, right = axs[0,2].get_xlim()
            bottom, top = axs[0,2].get_ylim()    
            xx = np.linspace(left, right, 100)
            yy = np.linspace(bottom, top, 100)
            grid = np.moveaxis(np.stack(np.meshgrid(xx, yy, indexing='xy')), 0, -1)
            extent = (left, right, bottom, top)    
    
            for ii in range(args.fit_gaussians):
                pdf = mvn.pdf(grid, mean=fit_means[ii], cov=fit_covs[ii])
                axs[0,2].contour(xx, yy, pdf, origin='lower', extent=extent, alpha=0.5, colors=[DARK[ii]])
                axs[0,2].scatter(fit_means[ii][0], fit_means[ii][1], s=100, color=DARK[ii], marker='x')   

            axs[0,2].set_title(f'Fit to {args.fit_gaussians} Gaussians')
            axs[0,2].set_xlabel('$x_1$')
            axs[0,2].set_ylabel('$x_2$')
    
    # Questions 3-4: Hidden Markov Models
    
    initial_probs = rng.random(args.num_hidden)
    transitions = rng.random((args.num_hidden, args.num_hidden)) + np.eye(args.num_hidden) * args.dwell
    initial_probs /= np.sum(initial_probs)
    for ii in range(args.num_hidden):
        transitions[ii,:] /= np.sum(transitions[ii,:])
    
    emission_means = np.arange(args.num_hidden)
    emission_sds = rng.random(args.num_hidden) * args.noise
    
    print(f'Q3: attempting to generate HMM sequence of {args.num_samples} points with {args.num_hidden} hidden states')
    
    x, z = generate_hmm_sequence ( args.num_samples, initial_probs, transitions,
                                   emission_means, emission_sds, rng )
    
    if x is None:
        print('-> not implemented')
        utils.plot_unimplemented(axs[1,0], title=f'Hidden Markov Model')
        utils.plot_unimplemented(axs[1,1], title=f'Viterbi Decoding (True Model)')
        utils.plot_unimplemented(axs[1,2], title=f'Viterbi Decoding (Fitted Model)')
    else:    
        axs[1,0].plot(x, color=DARK[1], label='Observed')
        axs[1,0].plot(z, color=DARK[0], label='Hidden')
        axs[1,0].set_title(f'Hidden Markov Model')
        axs[1,0].set_xlabel('Time')
        axs[1,0].set_ylabel('Value')
        axs[1,0].legend(loc='upper right')

        print('Q4: attempting to decode hidden states with true params')
        fit_z = viterbi ( x, initial_probs, transitions, emission_means, emission_sds )
        
        if fit_z is None:
            print('-> not implemented')
            utils.plot_unimplemented(axs[1,1], title=f'Viterbi Decoding (True Model)')
        else:
            axs[1,1].plot(z, color=DARK[1], label='True')
            axs[1,1].plot(fit_z, color=DARK[0], label='Decoded')
            axs[1,1].set_title(f'Viterbi Decoding (True Model)')
            axs[1,1].set_xlabel('Time')
            axs[1,1].set_ylabel('State')
            axs[1,1].legend(loc='upper right')

        if hmm is None:
            print('hmmlearn not available, skipping fit test')
            utils.plot_unimplemented(axs[1,2], 'Viterbi Decoding (hmmlearn fit)', msg='hmm learn unavailable')
        else:
            print('fitting model parameters from sequence with hmmlearn')
            hmg = hmm.GaussianHMM(args.num_hidden)
            hmg.fit(x.reshape(-1,1))
            fit_z = hmg.predict(x.reshape(-1,1))

            axs[1,2].plot(z, color=DARK[1], label='True')
            axs[1,2].plot(fit_z.ravel(), color=DARK[0], label='Decoded')
            axs[1,2].set_title(f'Viterbi Decoding (Fitted Model)')
            axs[1,2].set_xlabel('Time')
            axs[1,2].set_ylabel('State')
            axs[1,2].legend(loc='upper right')
    
    fig.tight_layout(pad=1)
    fig.savefig(args.file)
    plt.close(fig)
