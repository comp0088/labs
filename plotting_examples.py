#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The Matplotlib library provides extensive plotting capabilities,
originally modelled on those in Matlab. The library is big and
complicated, and though there is extensive documentation online
(see https://matplotlib.org) it can be quite daunting at first.

Here we provide a few simple example plots that you can use as
starting points for some of the visualisations in the lab
assignments. They are not intended to be exhaustive, nor to be
perfect exemplars of best practice. If you are already familiar
with Matplotlib or just want to dig into it on your own, feel free
to ignore these examples.
"""

import sys, os, os.path
import argparse

import numpy as np
import numpy.random
import matplotlib.pyplot as plt

rng = numpy.random.default_rng()

def histogram ( axes ):
    # generate some normally-distributed data
    xx = rng.normal(size=1000)
    
    # plot as a density histogram (ie, area sums to 1)
    axes.hist(xx, bins=21, density=True, color='cornflowerblue', edgecolor='black')
    
    axes.set_xlabel('Sample Value')
    axes.set_ylabel('Density')
    axes.set_title('Histogram of 1000 N(0,1) samples')


def xy_scatter ( axes ):
    # generate a noiseless ideal cubic curve
    xx = np.linspace(-5, 5, 51)
    yy = xx * xx * xx
    
    # you can control various aspects of the appearance when plotting
    # and you can also add a label to be used when you add a legend
    axes.plot(xx, yy, color='teal', marker='', linestyle='-', label='Ideal Curve')
    
    # generate some noisy sample points
    xx = rng.random(50) * 10 - 5
    yy = xx * xx * xx + rng.normal(scale=4, size=50)
    
    axes.plot(xx, yy, color='darkorchid', marker='o', linestyle='', label='Noisy Sample', alpha=0.5)

    # you can use (some) LaTeX formatting in text labels
    axes.set_xlabel('$x$')    
    axes.set_ylabel('$y = x^3$')
    
    # add a legend
    # by default this includes stuff we've given a label above
    axes.legend()
    
    axes.set_title('Simple Cubic Function')
    

def xyz_as_image ( axes, xx, yy, zz ):
    # plot the grid of zz data in 2d, using colour to represent height
    
    # the most common case of showing an image just uses the pixel dimensions for the axes
    
    # but for demo purposes here we actually want to match the ranges of x and y
    # and, just to be annoying, have also chosen unequal scales
    # so we need to specify both the extent (axis range occupied)
    # and the aspect ratio (ie, ratio of the sides)
    extent = (np.min(xx), np.max(xx), np.min(yy), np.max(yy))    
    aspect = (extent[1] - extent[0])/(extent[3] - extent[2])

    # images also usually draw from the top down
    # specifying origin='lower' here makes it draw upwards, like a graph 
    axes.imshow(zz, cmap='cividis', origin='lower', extent=extent, aspect=aspect)
    
    # add some contour lines
    levels = np.linspace(np.min(zz), np.max(zz), 7)
    axes.contour(xx, yy, zz, levels, cmap='rainbow', origin='lower', extent=extent, alpha=0.5 )
    
    axes.set_xlabel('$x$')
    axes.set_ylabel('$y$')
    axes.set_title('$z = \cos((x-1)^2 + (y+0.5)^2)$')


def xyz_as_surface ( axes, xx, yy, zz ):
    # plot the grid of zz data in a 3d projection
    axes.plot_surface( xx, yy, zz, rcount=100, ccount=100, cmap='ocean', antialiased=False, linewidth=0 )
    axes.set_zlim(-1.2, 1.2)
    
    axes.set_xlabel('$x$')
    axes.set_ylabel('$y$')
    axes.set_zlabel('$z$')
    axes.set_title('$z = \cos((x-1)^2 + (y+0.5)^2)$')

    
def make_xyz ():
    # generate a simple 3d dataset where z is a function of x and y
    x = np.linspace(-2.5, 2.5, 100)
    y = np.linspace(-1.5, 1.5, 100)
    xx, yy = np.meshgrid(x, y)
    zz = np.cos((xx-1)*(xx-1) + (yy+0.5)*(yy+0.5))
    return xx, yy, zz
    

if __name__ == '__main__':
    # create a figure to hold our examples
    fig = plt.figure(figsize=(8, 8))
        
    # add subplots to the figure and call the example plotting functions on them
    histogram ( fig.add_subplot(2, 2, 1) )
    xy_scatter ( fig.add_subplot(2, 2, 2) )
    
    # we'll use the same data for the next two
    xx, yy, zz = make_xyz ()
    
    xyz_as_image ( fig.add_subplot(2, 2, 3), xx, yy, zz )
    
    # specify projection here to get a 3d axis
    xyz_as_surface ( fig.add_subplot(2, 2, 4, projection='3d'), xx, yy, zz )
    
    # adjust the spacing around the subplots, to look better
    # spacing can be a bit iffy for 3d plots, so padding is relatively high
    # when using all 2d you can usually make it tighter
    fig.tight_layout(pad=4)
    
    # save to a PDF (other formats are also possible)
    fig.savefig('plotting_examples.pdf')
    
    # close the figure
    # in this case the script is about to exit, so it doesn't matter
    # but in general you should dispose of plotting objects once you're done with them
    plt.close(fig)
