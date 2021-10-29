#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
COMP0088 lab exercises for week 5.

Add your code as specified below.

A simple test driver is included in this script. Call it at the command line like this:

  $ python week_5.py

A 6-panel figure, `week_5.pdf`, will be generated so you can check it's doing what you
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

import utils


#### ADD YOUR CODE BELOW

# -- Question 1 --

def relu ( z ):
    """
    Rectified linear unit activation function.
    
    # Arguments
        z: a single number or numpy array
    
    # Returns
        r: a number or numpy array of the same dimensions
            as the input value, giving the ReLU of
            of each input value
    """
    # TODO: implement this
    return None

def d_relu ( z ):
    """
    Gradient of the ReLU function
    
    # Arguments
        z: a single number or numpy array
    
    # Returns
        r: a number or numpy array of the same dimensions
            as the input value, giving the gradient
            of the ReLU function at each input value
    """
    # TODO: implement this
    return None

def sigmoid ( z ):
    """
    Sigmoid activation function.
    
    # Arguments
        z: a single number or numpy array
    
    # Returns
        r: a number or numpy array of the same dimensions
            as the input value, giving the sigmoid (logistic)
            output for each input value
    """
    # TODO: implement this
    return None

def d_sigmoid ( z ):
    """
    Gradient of the sigmoid function
    
    # Arguments
        z: a single number or numpy array
    
    # Returns
        r: a number or numpy array of the same dimensions
            as the input value, giving the gradient
            of the sigmoid function at each input value
    """
    # TODO: implement this
    return None

def binary_crossentropy_loss ( y, y_hat, eps=1e-10 ):
    """
    Binary cross-entropy loss for predictions, given the
    true values.
    
    # Arguments:
        y: a numpy array of true binary labels.
        y_hat: a numpy array of predicted labels,
            as numbers in open interval (0, 1). must have
            the same number of entries as y, but not
            necessarily identical shape
        eps: a small offset to avoid numerical problems
            when predictions are very close to 0 or 1
    
    # Returns:
        loss: a numpy array of individual cross-entropy
            loss values for each prediction. will be
            the same shape as y_hat irrespective of the
            shape of y
    """
    # TODO: implement this
    return None

def d_binary_crossentropy_loss ( y, y_hat, eps=1e-10 ):
    """
    Gradient of the cross-entropy loss for predictions, given the
    true values.
    
    # Arguments:
        y: a numpy array of true binary labels.
        y_hat: a numpy array of predicted labels,
            as numbers in open interval (0, 1). must have
            the same number of entries as y, but not
            necessarily identical shape
        eps: a small offset to avoid numerical problems
            when predictions are very close to 0 or 1
    
    # Returns:
        grad: a numpy array of individual cross-entropy
            gradient values for each prediction. will be
            the same shape as y_hat irrespective of the
            shape of y
    """
    # TODO: implement this
    return None


# -- Question 2 --

def init_layer ( fan_in, fan_out, act, rng ):
    """
    Create a single neural network layer.
    
    # Arguments
        fan_in: the number of incoming connections
        fan_out: the number of outgoing connections
        act: name of the activation function for the
            layer, either "sigmoid" or "relu"
        rng: an instance of numpy.random.Generator
            from which to draw random numbers
    
    # Returns
        layer: a dict holding the layer contents, with
            keys 'W', 'b', 'shape' and 'act'.
            (See the coursework for full details.)
    """
    # TODO: implement this
    return None


def init_mlp ( spec, rng ):
    """
    Build a neural network according to the
    given specification.
    
    # Arguments
        spec: an iterable of tuples (fan_in, act)
            specifying the configuration of the network layers.
            there must be at least 2 elements; the last is only
            used to determine output size of the layer before,
            it does not create a layer of its own
        rng: an instance of numpy.random.Generator
            from which to draw random numbers
    
    # Returns
        mlp: a list of layer dicts
    """
    assert(len(spec) > 1)
    
    # TODO: implement this
    return None


# -- Question 3 --

def layer_forward ( layer, X ):
    """
    Run a forward pass of data through a layer, storing
    intermediate values in the layer dict.
    
    # Arguments
        layer: a layer dict as created by init_layer
        X: the input data to the layer, a matrix
            where the columns are features and the
            rows are samples. feature count must
            match the layer's fan_in
    
    # Returns
        A: the layer's output activations, a matrix where the
            columns are (fan_out) features and the rows are
            samples
    """
    assert(X.shape[-1] == layer['W'].shape[0])

    # TODO: implement this
    return None


def mlp_forward ( mlp, X ):
    """
    Run a forward pass through a whole neural net.
    
    # Arguments
        mlp: a list of layer dicts, as created by init_mlp
        X: the input data to the network, a matrix
            where the columns are features and the
            rows are samples. feature count must
            match the first layer's fan_in
    
    # Returns
        A: the output activations of the final network layer
    """
    # TODO: implement this
    return None



# -- Question 4 --

def layer_backward ( layer, dA ):
    """
    Run a backward pass of gradients through a layer, storing
    computed values in the layer dict. The forward pass must
    have been performed first.
    
    # Arguments
        layer: a layer dict as created by init_layer
        dA: the gradients of the loss with respect to the
            forward pass activations. a matrix the same shape
            as those previously computed activations.
    
    # Returns
        dX: the gradients of the loss with respect to the
            layer inputs from the forward pass
    """
    assert(dA.shape == layer['A'].shape)
    
    # TODO: implement this
    return None



def mlp_backward ( mlp, d_loss ):
    """
    Backpropagate gradients through the whole neural net.
    The forward pass must have been performed first.
    
    # Arguments
        mlp: a list of layer dicts, as created by init_mlp
        d_loss: the gradients of the loss at the final
            layer output, a matrix the same shape
            as previously computed activations.
    
    # Returns
        None
    """
    # TODO: implement this
    pass


# -- Question 5 --

def layer_update ( layer, lr ):
    """
    Update layer weights & biases according to the previously
    computed gradients. Forward and backward passes
    must both have been performed.
    
    # Arguments
        layer: a layer dict as created by init_layer
        lr: the learning rate
    
    # Returns
        None
    """
    # TODO: implement this
    pass

    

def mlp_update ( mlp, lr ):
    """
    Update all network weights & biases according to the
    previously computed gradients. Forward and backward passes
    must both have been performed.
    
    # Arguments
        mlp: a list of layer dicts, as created by init_mlp
        lr: the learning rate
    
    # Returns
        None
    """
    # TODO: implement this
    pass


# -- Question 6 --

def mlp_minibatch ( mlp, X, y, lr ):
    """
    Fit a neural network to a single mini-batch
    of training data.
    
    # Arguments
        mlp: a list of layer dicts, as created by init_mlp
        X: an array of sample data, where rows are samples
            and columns are features. feature dimension must
            match the input dimension of mlp.
        y: vector of binary class labels corresponding to the
            samples, must be same length as number of rows in X
        lr: the learning rate
    
    # Returns
        loss: the mean training loss over the minibatch
    """
    assert(X.shape[0] == len(y))
    assert(X.shape[-1] == mlp[0]['W'].shape[0])
    
    # TODO: implement this
    return None


def mlp_epoch ( mlp, X, y, batch, lr, rng ):
    """
    Fit a neural network for one epoch -- ie, a single
    pass through the data in minibatches of specified size.
    
    # Arguments
        mlp: a list of layer dicts, as created by init_mlp
        X: an array of sample data, where rows are samples
            and columns are features. feature dimension must
            match the input dimension of mlp.
        y: vector of binary class labels corresponding to the
            samples, must be same length as number of rows in X
        batch: the size of minibatches to train
        lr: the learning rate
        rng: an instance of numpy.random.Generator
            from which to draw random numbers
    
    # Returns
        loss: the mean training loss over the whole dataset    
    """
    # TODO: implement this
    return None


def mlp_train ( mlp, X, y, batch, epochs, lr, rng ):
    """
    Fit a neural network iteratively for multiple
    epochs.

    # Arguments
        mlp: a list of layer dicts, as created by init_mlp
        X: an array of sample data, where rows are samples
            and columns are features. feature dimension must
            match the input dimension of mlp.
        y: vector of binary class labels corresponding to the
            samples, must be same length as number of rows in X
        batch: the size of minibatches to train
        epochs: number of epochs to train
        lr: the learning rate
        rng: an instance of numpy.random.Generator
            from which to draw random numbers
    
    # Returns
        loss: a list of the mean training loss at each epoch    
    """
    # TODO: implement this
    return None



# -- Question 7 --

def mlp_predict ( mlp, X, thresh=0.5 ):
    """
    Make class predictions from a neural network.

    # Arguments
        mlp: a list of layer dicts, as created by init_mlp
        X: an array of test data, where rows are samples
            and columns are features. feature dimension must
            match the input dimension of mlp.
        thresh: the decision threshold
        
    # Returns
        y_hat: a vector of predicted binary labels for X
    """
    # TODO: implement this
    return None


#### TEST DRIVER

def process_args():
    ap = argparse.ArgumentParser(description='week 5 coursework script for COMP0088')
    ap.add_argument('-s', '--seed', help='seed random number generator', type=int, default=None)
    ap.add_argument('-n', '--num_samples', help='number of samples to use', type=int, default=50)
    ap.add_argument('-r', '--resolution', help='grid sampling resolution for classification plots', type=int, default=20)
    ap.add_argument('-l', '--lr', help='learning rate', type=float, default=1e-2)
    ap.add_argument('-e', '--epochs', help='number of epochs to train', default=100, type=int)
    ap.add_argument('-b', '--batch', help='minibatch size to train', default=1, type=int)
    ap.add_argument('-y', '--layers', help='comma-sep list of hidden layer sizes', default='4,3')
    ap.add_argument('-d', '--data', help='CSV file containing training data', default='week_3_data.csv')
    ap.add_argument('file', help='name of output file to produce', nargs='?', default='week_5.pdf')
    return ap.parse_args()


if __name__ == '__main__':
    args = process_args()
    rng = numpy.random.default_rng(args.seed)

    LIMITS=(-5, 5)
    hidden = [ int(x) for x in args.layers.split(',') ]
    layers=((2, 'relu'),) + tuple((int(x), 'relu') for x in hidden[:-1]) + ((hidden[-1], 'sigmoid'), (1, None))

    if True:
        print(f'loading data from {args.data}')
        df = pd.read_csv(args.data)
        X = df[['X1','X2']].values[:args.num_samples,:]
        y = df['Binary'].values[:args.num_samples]
    else:
        X, y = utils.random_sample(lambda z: z[:,1] > 0, num_samples=args.num_samples, limits=LIMITS)

    fig = plt.figure(figsize=(12, 8))
    axs = fig.subplots(nrows=2, ncols=3)
    blue, orange = plt.cm.tab10.colors[:2]
    
    xx = np.linspace(-5,5,100)
    yy = np.linspace(0,1,100)
    y0 = np.zeros(100)
    y1 = np.ones(100)

    print('Q1: plotting relu')
    aa = relu(xx)
    dd = d_relu(xx)
    if (aa is None) and (dd is None):
        utils.plot_unimplemented(axs[0,0], 'ReLU')
    else:
        if aa is not None: axs[0,0].plot(xx, aa, color=blue, label='Activation')
        if dd is not None: axs[0,0].plot(xx, dd, color=orange, linestyle='dashed', label='Gradient')
        axs[0,0].set_title('ReLU')
        axs[0,0].set_xlabel('$x$')
        axs[0,0].set_ylabel('$f$')
        axs[0,0].legend()
    
    print('Q1: plotting sigmoid')
    aa = sigmoid(xx)
    dd = d_sigmoid(xx)
    if (aa is None) and (dd is None):
        utils.plot_unimplemented(axs[0,1], 'Sigmoid')
    else:
        if aa is not None: axs[0,1].plot(xx, aa, color=blue, label='Activation')
        if dd is not None: axs[0,1].plot(xx, dd, color=orange, linestyle='dashed', label='Gradient')
        axs[0,1].set_title('Sigmoid')
        axs[0,1].set_xlabel('$x$')
        axs[0,1].set_ylabel('$f$')
        axs[0,1].legend()
    
    print('Q1: plotting binary cross-entropy')
    ll0 = binary_crossentropy_loss(y0, yy)
    ll1 = binary_crossentropy_loss(y1, yy)
    dd0 = d_binary_crossentropy_loss(y0, yy)
    dd1 = d_binary_crossentropy_loss(y1, yy)

    if (ll0 is None) and (ll1 is None) and (dd0 is None) and (dd1 is None):
        utils.plot_unimplemented(axs[0,2], 'Binary Cross-Entropy')
    else:        
        if ll0 is not None: axs[0,2].plot(yy, ll0, color=blue, label='Loss, $y=0$')
        if dd0 is not None: axs[0,2].plot(yy, dd0, color=blue, linestyle='dashed', label='Gradient, $y=0$')
        if ll1 is not None: axs[0,2].plot(yy, ll1, color=orange, label='Loss, $y=1$')
        if dd1 is not None: axs[0,2].plot(yy, dd1, color=orange, linestyle='dashed', label='Gradient, $y=1$')
        axs[0,2].set_title('Binary Cross-Entropy')
        axs[0,2].set_xlabel('$\\hat{y}$')
        axs[0,2].set_ylabel('$f$')
        axs[0,2].set_ylim(-35,35)
        axs[0,2].legend(loc='lower right')

    print(f'Q2: initialising MLP with {len(layers)-1} layers')
    nn = init_mlp ( layers, rng )
    if nn is None:
        print('Not implemented')
        utils.plot_unimplemented(axs[1,0], 'Training Loss')
        utils.plot_unimplemented(axs[1,1], 'Neural Network (MLP)')
    else:
        pprint.pprint([ (layer['shape'], layer['act']) for layer in nn ])
        print(f'Q3-6: training MLP for {args.epochs} epochs')
        t0 = perf_counter()
        loss = mlp_train ( nn, X, y, args.batch, args.epochs, args.lr, rng )
        if loss is None:
            print('Not implemented')
            utils.plot_unimplemented(axs[1,0], 'Training Loss')
            utils.plot_unimplemented(axs[1,1], 'Neural Network (MLP)')
        else:
            print(f'time taken: {perf_counter()-t0:.2f} s')
            print('Q3-6: plotting loss')
            axs[1,0].plot(loss)
            axs[1,0].set_title('Training Loss')
            axs[1,0].set_xlabel('Epoch')
            axs[1,0].set_ylabel('Mean Binary Cross-Entropy')
            
            print('Q7: plotting fit')
            t0 = perf_counter()
    
            utils.plot_classification_map(axs[1,1], lambda z: mlp_predict(nn, z), X, y,
                                          resolution=args.resolution, limits=LIMITS,
                                          title=f'{len(layers)-1}-Layer MLP')
            print(f'time taken: {perf_counter()-t0:.2f} s')
            
            print('Q7: plotting output activations')
            im = nn[-1]['A'].reshape((args.resolution,args.resolution))
            axs[1,2].imshow(im.T, origin='lower', extent=LIMITS * 2, cmap='Greens')
            axs[1,2].set_title('Output Activations')
            axs[1,2].set_xlabel('$x_1$')
            axs[1,2].set_ylabel('$x_2$')
            
    fig.tight_layout(pad=1)
    fig.savefig(args.file)
    plt.close(fig)
