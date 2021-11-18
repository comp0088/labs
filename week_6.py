#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
COMP0088 lab exercises for week 6.

Add your code as specified below.

A simple test driver is included in this script. Call it at the command line like this:

  $ python week_6.py

A 3-panel figure, `week_6.pdf`, will be generated so you can check it's doing what you
want. You should not need to edit the driver code, though you can if you wish.
"""

import sys, os, os.path
import argparse
from time import perf_counter

import numpy as np
import numpy.random
import matplotlib
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import utils

# these will be extracted from data anyway, but
# these are the defaults for MNIST
INPUT_SHAPE = [1, 28, 28]
INPUT_SIZE = np.prod(INPUT_SHAPE)

# default layer configurations for the different model types
DEFAULT_MLP = [32, 32]
DEFAULT_CNN = [16, 16]
DEFAULT_RNN = [64, 1]

NUM_CLASSES = 10

# -- placeholder model class --

class Model(nn.Module):
    """
    Trivial PyTorch linear model to act as a placeholder
    and parent for the different task classes.
    """

    def __init__(self, input=INPUT_SIZE, output=NUM_CLASSES):
        """
        Initialise the model.
        
        The data passed to all the models this week is *image* data.
        Here we flatten it into a vector and apply a single linear
        layer with no activation.
        
        # Arguments
            input: the input size for the linear layer
            output: the output size for the linear layer
        """
        super().__init__()
        self.linear = nn.Sequential( nn.Flatten(), nn.Linear(input, output) )
    
    def forward(self, x):
        """
        Execute a forward pass of the model.
        
        # Arguments
            x: the input data to the first layer
        
        # Returns
            output tensor from the last layer
        """
        return self.linear(x)
    

#### ADD YOUR CODE BELOW

# -- Question 1 --

def train_epoch(model, dataloader, loss_function, optimiser):
    """
    Train a model on a single epoch of data.
    
    # Arguments
        model: a pytorch model (eg one of the above nn.Module subclasses)
        dataloader: a pytorch dataloader that will iterate over the dataset
            in batches
        loss_function: a pytorch loss function tensor
        optimiser: optimiser to use for training
    
    # Returns:
        loss: mean loss over the whole epoch
        accuracy: mean prediction accuracy over the epoch
    """
    # TODO: implement this
    return None, None


def test_epoch(model, dataloader, loss_function):
    """
    Evaluate a model on a dataset.
    
    # Arguments
        model: a pytorch model (eg one of the above nn.Module subclasses)
        dataloader: a pytorch dataloader that will iterate over the dataset
            in batches
        loss_function: a pytorch loss function tensor
    
    # Returns:
        loss: mean loss over the whole epoch
        accuracy: mean prediction accuracy over the epoch
    """
    # TODO: implement this
    return None, None


# -- Question 2 --

class MLP(Model):
    """
    Simple multi-layer perceptron model for image classification.
    """

    def __init__(self, input=INPUT_SIZE, spec=DEFAULT_MLP, output=NUM_CLASSES):
        """
        Initialise the multi-layer perceptron with the specified arrangement
        of fully-connected layers, using ReLU activation on each layer.
        
        Note that the data passed to this model (as for all models this
        week) will be *image* data, so the first layer in the model
        should be a Flatten layer, to turn the C * W * H pixel arrays
        into one dimensional ones.
        
        # Arguments
            input: the input size for the first hidden layer
            spec: a list of sizes for the intermediate layers
            output: the output size for the final layer
        """
        super().__init__(input, output)
        # TODO: implement this
    
    def forward(self, x):
        """
        Execute a forward pass of the model.
        
        # Arguments
            x: the input data to the first layer
        
        # Returns
            output tensor from the last layer
        """
        
        # TODO: remove line below and implement this
        return super().forward(x)


# -- Question 3 --

class CNN(Model):
    """
    Simple convolutional neural network model for image classification.
    """
    
    def __init__(self, input=INPUT_SHAPE, spec=DEFAULT_CNN, output=NUM_CLASSES,
                 kernel=3, stride=2, padding=1):
        """
        Initialise the CNN with the specified arrangement of 2D convolutional
        layers and ReLU activations, with a final fully-connected layer to
        do the classification.
        
        # Arguments
            input: the input shape for the first convolutional layer
            spec: a list of numbers of channels for the convolutional layers
            output: the output size for the final fully-connected layer
            kernel: kernel size for all layers
            stride: convolution stride for all layers
            padding: amount of padding to add around each layer before convolving
        """
        super().__init__(np.prod(input), output)
        assert(len(spec) > 0)
        # TODO: implement this
    
    def forward(self, x):
        """
        Execute a forward pass of the model.
        
        # Arguments
            x: the input data to the first layer
        
        # Returns
            output tensor from the last layer
        """
        # TODO: remove line below and implement this
        return super().forward(x)


# -- Question 4 --

class RNN(Model):
    """
    Simple recurrent network model for image classification.
    """

    def __init__(self, input=INPUT_SHAPE, spec=DEFAULT_RNN, output=NUM_CLASSES,
                 unit_type='lstm'):
        """
        Initialise the RNN with the specified stack of recurrent layers,
        with a final fully-connected layer to do the classification.
        
        # Arguments
            input: the input shape for the image data.
                we assume [C, H, W] ordering and will process as a
                sequence of H inputs of size C*W
            spec: a list specifying the hidden size of each layer (at element 0)
                and optionally the number of such layers to stack (at element 1)
            output: the output size for the final fully-connected layer
            unit_type: what type of recurrent layers to use
                'lstm': use nn.LSTM
                'gru': use nn.GRU
                (anything else): use nn.RNN
        """
        super().__init__(np.prod(input), output)
        # TODO: implement this
    
    def forward(self, X):
        """
        Execute a forward pass of the model.
        
        # Arguments
            x: the input data to the first layer
        
        # Returns
            output tensor from the last layer
        """
        # TODO: remove line below and implement this
        return super().forward(x)



#### TEST DRIVER

def subset_classes ( vis_data, args ):
    """
    Optionally reduce a torchvision dataset to a subset of its classes.
    Should allow for faster train & test cycles.
    
    Note that this depends on the dataset having labels in its
    `targets` attribute -- this is true of the 5 supported datasets
    here, but is not necessarily true of torch datasets in
    general. (For example, it is not true of a subsetted dataset
    returned from this function.)
    """
    if args.classes > 1:
        print(f'subsetting first {args.classes} classes')
        targets = np.array( vis_data.targets )
        return Subset(vis_data, np.flatnonzero(targets < args.classes))
    else:
        return vis_data


def process_args():
    ap = argparse.ArgumentParser(description='week 6 coursework script for COMP0088')
    ap.add_argument('-s', '--seed', help='seed random number generator', type=int, default=None)

    # model configuration
    ap.add_argument('-M', '--model', help='type of model to build', default='linear')
    ap.add_argument('-y', '--layers', help='comma-sep list of hidden layer sizes', default=None)
    ap.add_argument('-k', '--kernel', help='convolution kernel size', default=3, type=int)
    ap.add_argument('-S', '--stride', help='convolution stride', default=2, type=int)
    ap.add_argument('-p', '--padding', help='convolution padding', default=1, type=int)

    # training configuration
    ap.add_argument('-b', '--batch', help='minibatch size to train', default=25, type=int)
    ap.add_argument('-e', '--epochs', help='number of epochs to train', default=5, type=int)
    ap.add_argument('-l', '--lr', help='learning rate', type=float, default=1e-2)
    ap.add_argument('-o', '--optim', help='which optimiser to use', default='sgd')
    ap.add_argument('-m', '--momentum', help='learning momentum', default=0.9, type=float)
    ap.add_argument('-d', '--decay', help='weight decay (L2 regularisation)', default=0, type=float)
    ap.add_argument('-c', '--classes', help='how many classes to use', default=0, type=int)
    
    ap.add_argument('-C', '--cpu', help='use CPU even if CUDA available', action='store_true')
    
    # problem configuration
    ap.add_argument('-D', '--data', help='name of dataset to load', default='usps')

    ap.add_argument('file', help='name of output file to produce', nargs='?', default='week_6.pdf')
    return ap.parse_args()


if __name__ == '__main__':
    args = process_args()
    rng = numpy.random.default_rng(args.seed)
    if args.seed is not None: torch.manual_seed(args.seed)
    
    data, name = {
        'm' : (datasets.MNIST, 'MNIST'),
        'f' : (datasets.FashionMNIST, 'FashionMNIST'),
        'k' : (datasets.KMNIST, 'Kuzushiji-MNIST'),
        'c' : (datasets.CIFAR10, 'CIFAR10'),
    }.get(args.data.lower()[:1], (datasets.USPS, 'USPS'))
    
    t0 = perf_counter()
    print('loading training data for ' + name)
    train_data = subset_classes(data('data', train=True, download=True, transform=ToTensor()), args)
    train_loader = DataLoader(train_data, batch_size=args.batch)

    print('loading validation data for ' + name)
    t0 = perf_counter()
    val_data = subset_classes(data('data', train=False, download=True, transform=ToTensor()), args)
    val_loader = DataLoader(val_data, batch_size=args.batch)

    print(f'loading time: {perf_counter() - t0:>.3f} seconds')
    
    input_shape = tuple(train_data[0][0].shape)
    input_size = np.product(input_shape)
    
    device = 'cuda' if (torch.cuda.is_available() and not args.cpu) else 'cpu'
    print('using device: ' + device)
    
    if args.layers is None:
        args.layers = {
            'mlp' : ','.join([str(x) for x in DEFAULT_MLP]),
            'cnn' : ','.join([str(x) for x in DEFAULT_CNN]),
        }.get(args.model.lower(), ','.join([str(x) for x in DEFAULT_RNN]))

    layers = [ int(x) for x in args.layers.split(',') ]
    
    t1 = perf_counter()
    num_classes = min(args.classes, NUM_CLASSES) if args.classes > 1 else NUM_CLASSES
    
    if args.model.lower() == 'mlp':
        print('building MLP with hidden layers: ' + str(layers))
        model = MLP(input=input_size, spec=layers, output=num_classes).to(device)
        model_name = 'MLP ' + str(layers)
    elif args.model.lower() == 'cnn':
        print('building CNN with layers: ' + str(layers))
        model = CNN(input=input_shape, spec=layers, output=num_classes,
                    kernel=args.kernel, stride=args.stride, padding=args.padding).to(device)
        model_name = 'CNN ' + str(layers)
    elif args.model.lower() == 'rnn':
        print('building RNN with layers: ' + str(layers))
        model = RNN(input=input_shape, spec=layers, output=num_classes, unit_type='rnn').to(device)
        model_name = 'RNN ' + str(layers)
    elif args.model.lower() == 'lstm':
        print('building LSTM with layers: ' + str(layers))
        model = RNN(input=input_shape, spec=layers, output=num_classes, unit_type='lstm').to(device)
        model_name = 'LSTM ' + str(layers)
    elif args.model.lower() == 'gru':
        print('building GRU with layers: ' + str(layers))
        model = RNN(input=input_shape, spec=layers, output=num_classes, unit_type='gru').to(device)
        model_name = 'GRU ' + str(layers)
    else:
        print(f'building placeholder linear model {input_size}  {num_classes}')
        model = Model(input=input_size, output=num_classes).to(device)
        model_name = f'Linear [{input_size}×{num_classes}]'
    
    print(f'build time: {perf_counter() - t1:>.3f} seconds')
    
    loss_function = nn.CrossEntropyLoss()
    
    if args.optim.lower() == 'adam':
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    # ... other optimisers here, maybe
    else:
        optimiser = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
    
    t2 = perf_counter()
    
    print(f'training for {args.epochs} epochs\n')
    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    for ii in range(args.epochs):
        print(f'Epoch {ii+1}\n-----------------------')
        loss, acc = train_epoch(model, train_loader, loss_function, optimiser)
        train_loss.append(loss)
        train_acc.append(acc)
        
        loss, acc = test_epoch(model, val_loader, loss_function)
        val_loss.append(loss)
        val_acc.append(acc)
    
    print(f'Total train/test time for {args.epochs} epochs: {perf_counter() - t2:>.2f} seconds')
    
    fig = plt.figure(figsize=(12, 4))
    axs = fig.subplots(nrows=1, ncols=3, squeeze=False)
    
    blue, orange = plt.cm.tab10.colors[:2]
    
    sub_tag = f' [{args.classes}]' if args.classes > 1 else ''

    utils.plot_image(axs[0,0], utils.torch_data_to_image_grid(train_data, 8, 8, shuffle=True, rng=rng),
                     title=f'{name}{sub_tag} Data Examples')
    
    if train_loss[0] is None:
        utils.plot_unimplemented(axs[0,1], title=f'Loss - {model_name}')
        utils.plot_unimplemented(axs[0,2], title=f'Accuracy - {model_name}')
    else:
        epochs = list(range(1, args.epochs+1))
        axs[0,1].plot(epochs, train_loss, color=blue, label='Training')
        axs[0,1].plot(epochs, val_loss, color=orange, label='Validation')
        axs[0,1].set_xlabel('Epoch')
        axs[0,1].set_ylabel('Loss')
        axs[0,1].set_title(f'Loss - {model_name}')
        axs[0,1].legend()
    
        axs[0,2].plot(epochs, train_acc, color=blue, label='Training')
        axs[0,2].plot(epochs, val_acc, color=orange, label='Validation')
        axs[0,2].set_xlabel('Epoch')
        axs[0,2].set_ylabel('Accuracy')
        axs[0,2].set_title(f'Accuracy - {model_name}')
        axs[0,2].legend()
    
    fig.tight_layout(pad=1)
    fig.savefig(args.file)
    plt.close(fig)
