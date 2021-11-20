#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
COMP0088 lab exercises for week 7.

Add your code as specified below.

A simple test driver is included in this script. Call it at the command line like this:

  $ python week_7.py

You should not need to edit the driver code, though you can if you wish.
"""
import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import distance

##### !!! DO NOT CHANGE THESE !!!
GAUSSIAN_DIM = 2
N_PCA = 2
#####


def get_colour_set(n_colours, rng):
    """
    Generates a (unique) list of randomly generated RGB colours.

    # Arguments:
        n_colours: the number of colours to generate.
        rng: an instance of numpy.random.Generator.

    # Returns:
        colours: a list of randomly generated colours.
    """
    colours = []

    for _ in range(n_colours):
        cand = None

        while cand is None or cand in colours:
            cand = (rng.uniform(), rng.uniform(), rng.uniform())

        colours.append(cand)

    return colours


def generate_gaussian_samples(mean_shifts, n_samples=100):
    """
    Generates samples from multiple multivariate Gaussian distributions.

    # Arguments:
        mean_shifts: list of numbers, where each represents
            the mean shift from a multivariate standard normal.
        len(mean_shifts) + 1 represents the number of Gaussians
            to draw samples from.
        n_samples: the number of samples to draw from each Gaussian.

    # Returns:
       data: sampled data of shape (len(mean_shifts) + 1, GAUSSIAN_DIM).
    """
    data = np.random.multivariate_normal(
        mean=np.zeros(GAUSSIAN_DIM),
        cov=np.eye(GAUSSIAN_DIM, GAUSSIAN_DIM),
        size=n_samples,
    )

    for m_shift in mean_shifts:
        data = np.concatenate(
            (
                data,
                np.random.multivariate_normal(
                    mean=np.zeros(GAUSSIAN_DIM) + m_shift,
                    cov=np.eye(GAUSSIAN_DIM, GAUSSIAN_DIM),
                    size=n_samples,
                ),
            ),
            axis=0,
        )

    return data


def plot_gaussian_data(axes, data, rng, n_samples, n_gaussians):
    """
    Plots a list of multivariate Gaussians.

    # Arguments:
        axes: a Matplotlib Axes object into which to plot.
        data: data of shape (N, D) to plot.
        rng: an instance of numpy.random.Generator.
        n_samples: the number of samples to draw from each Gaussian.
    """
    colours = get_colour_set(n_gaussians, rng)

    for i in range(n_gaussians):
        start, end = i * n_samples, (i + 1) * n_samples
        axes.scatter(
            data[start:end, 0],
            data[start:end, 1],
            color=colours[i],
        )

    axes.set_xlabel("$x_1$")
    axes.set_ylabel("$x_2$")
    axes.set_title(f"Data sampled from {n_gaussians} Gaussians", fontsize=14)
    axes.grid()


def plot_k_means(axes, data, labels, rng, k):
    """
    Plots the converged k-means clustering assignment.

    # Arguments:
        axes: a Matplotlib Axes object into which to plot.
        data: input data of shape (N, D).
        labels: labels corresponding to input data of shape (N).
        rng: an instance of numpy.random.Generator.
    """
    colours = get_colour_set(len(np.unique(labels)), rng)

    for i, label in enumerate(np.unique(labels)):
        data_sub = data[labels == label]
        axes.scatter(
            data_sub[:, 0],
            data_sub[:, 1],
            color=colours[i],
        )

    axes.set_xlabel("$x_1$")
    axes.set_ylabel("$x_2$")
    axes.set_title(f"k-means clustering (k={k})", fontsize=14)
    axes.grid()


def compute_wcss(data, centroids, assignments):
    """
    Compute the within cluster sum of squares (WCSS).

    # Arguments:
        data: input data of shape (N, D).
        centroids: array of shape (k, D) containing the centroids for
            each cluster.
        assignments: array of shape (N) containing the assigned clusters
            for each row in data.

    # Returns:
        wcss: the WCSS across all clusters.
    """
    # TODO: implement the within cluster sum of squares
    return None


def k_means_clustering(data, rng, k=3, init="forgy", max_iter=500):
    """
    Method for implementing k-means clustering.

    # Arguments:
        data: input data of shape (N, D).
        rng: an instance of numpy.random.Generator.
        k: the number k for the algorithm.
        init: the initialisation method (one of "forgy", "naive").
        max_iter: the maximum number of iterations to run the algorithm.

    # Returns:
        assignments: list of shape (N) of cluster assignments
            for each data point.
        n_iter: the number of iterations needed by the algorithm.
        centroids: the centroids after convergence.
    """
    # TODO: implement the k-means clustering algorithm
    return None, None, None


def image_colour_quantisation(img, k, max_iter, rng):
    """
    Perform image colour quantisation on the provided input image.

    # Arguments:
        img: an array of shape (N, M, C) with N x M pixels and
            C colour channels per pixel.
        k: the number of clusters k to use for k-means.
        max_iter: the maximum number of iterations to run the
            clustering algorithm for.
        rng: an instance of numpy.random.Generator.

    # Returns:
        img_seq: an array of shape (N * M, C) representing the
            sequentialised image after the colour channels C have
            been assigned to the k clusters using the k-means algorithm.
    """
    # TODO: implement image colour quantisation using the k-means algorithm.
    return None


def load_word_embeddings(path):
    """
    Helper function to load word embeddings.

    # Arguments:
        path: the path to the word emebedings file (embeds.txt).

    # Returns:
        words: a list of words for the corresponding embeddings.
        embeddings: array of embeddings of shape (len(words), 300).
    """
    words, embeddings = [], []

    with open(path, "r") as f:
        data = f.readlines()
        for line in data:
            line_spl = line.split(" ")
            words.append(line_spl[0])
            embeddings.append([float(x) for x in line_spl[1:]])

    return words, np.array(embeddings)


def get_nearest_neighbours(query, embeddings, labels, n=10):
    """
    Computes nearest neighbours (in terms of Cosine similarity) for
        a given input query in embedding space.

    # Arguments:
        query: the query word.
        embeddings: list of word embeddings of shape (N, D).
        labels: list of N labels for the word embeddings.
        n: number of nearest neighbours to return.

    # Returns:
        nearest: list of words representing the n nearest neighbours
            for query (INCLUDING query itself).
    """
    assert query in labels, print(f"No embedding for word {query}")
    # TODO: find the nearest neighbours for word 'query' in 'embeddings'
    return None


def pca_eigen(data, n):
    """
    Computes PCA by eigendecomposition.

    # Arguments:
        data: input data of shape (N, D).
        n: number of principal components to retain.

    # Returns:
        The transformed data, truncated at n along the
            second dimension (shape (N, n)).
    """
    # TODO: transform the input data using PCA and
    # return only the n principal components
    return None


def pca_svd(data, n):
    """
    Computes PCA by singular value decomposition.

    # Arguments:
        data: input data of shape (N, D).
        n: number of principal components to retain.

    # Returns:
        The transformed data, truncated at n along the
            second dimension (shape (N, n)).
    """
    # TODO: transform the input data using PCA and
    # return only the n principal components
    return None


def pca(data, method="eigen", n_pcs=2):
    """
    Wrapper method for PCA implementations.

    # Arguments:
        data: input data of shape (N, D).
        method: the PCA method to use (one of "eigen", "svd").
        n_pcs: the number of principal components to retain.

    # Returns:
        data: the transformed data.
    """
    data = data - np.mean(data, axis=0)

    if method == "eigen":
        data = pca_eigen(data, n=n_pcs)
    elif method == "svd":
        data = pca_svd(data, n=n_pcs)
    else:
        print(f"Wrong PCA method {method} specified.")
        exit()

    return data


def plot_pca(axes, data, labels, selection, rng, method, n):
    """
    Plots data after PCA transformation.

    # Arguments:
        axes: a Matplotlib Axes object into which to plot
        data: input data of shape (N, n).
        labels: a list of N labels for samples in data.
        selection: dict where keys are words and values are lists
            of their nearest neighbours in embedding space.
        rng: an instance of numpy.random.Generator.
        method: the used PCA method.
        n: the number n of principal components retained.
    """
    colours = get_colour_set(len(selection.keys()), rng)

    for i, (_, neighbours) in enumerate(selection.items()):
        for j in [labels.index(x) for x in neighbours]:
            axes.scatter(data[j, 0], data[j, 1], color=colours[i])
            axes.annotate(labels[j], (data[j, 0], data[j, 1]))

    axes.set_xlabel("$x_1$")
    axes.set_ylabel("$x_2$")
    axes.set_title(f"PCA with {method} (n={n})", fontsize=14)
    axes.grid()


def process_args():
    ap = argparse.ArgumentParser(description="week 7 coursework script for COMP0088")

    ap.add_argument(
        "--n_samples",
        dest="n_samples",
        help="number of samples to draw per Gaussian",
        nargs="?",
        type=int,
        default=500,
    )

    ap.add_argument(
        "--n_clusters",
        dest="n_clusters",
        help="number of clusters to use for k-means",
        nargs="?",
        type=int,
        default=3,
    )

    ap.add_argument(
        "--n_clusters_q",
        dest="n_clusters_q",
        help="number of clusters to use for image quantisation k-means",
        nargs="?",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--k_means_init",
        dest="k_means_init",
        help="the initialisation method for k-means (one of 'forgy', 'naive')",
        nargs="?",
        type=str,
        default="forgy",
    )

    ap.add_argument(
        "--max_iter",
        dest="max_iter",
        help="maximum number of iterations for k-means algorithm",
        nargs="?",
        type=int,
        default=100,
    )

    ap.add_argument(
        "--max_iter_q",
        dest="max_iter_q",
        help="maximum number of iterations for image quantisation k-means algorithm",
        nargs="?",
        type=int,
        default=20,
    )

    ap.add_argument(
        "--mean_shifts",
        dest="mean_shifts",
        help="list of floats (comma-separated string) denoting mean shifts for the Gaussians to draw from",
        nargs="+",
        type=str,
        default=[-1.5, 1.5],
    )

    ap.add_argument(
        "--file",
        dest="file",
        help="name of output file to produce",
        nargs="?",
        type=str,
        default="week_7.pdf",
    )

    return ap.parse_args()


if __name__ == "__main__":
    args = process_args()
    rng = np.random.default_rng()
    fig = plt.figure(figsize=(12, 12))
    axs = fig.subplots(nrows=3, ncols=2, squeeze=False)

    mean_shifts = [float(s) for s in args.mean_shifts]
    n_gaussians = len(mean_shifts) + 1

    k_means_data = generate_gaussian_samples(
        mean_shifts=mean_shifts, n_samples=args.n_samples
    )

    plot_gaussian_data(axs[0, 0], k_means_data, rng, args.n_samples, n_gaussians)

    print(
        f"k-means clustering (k={args.n_clusters}) with {args.k_means_init} initialisation"
    )
    assignments, n_iter, _ = k_means_clustering(
        k_means_data,
        rng,
        k=args.n_clusters,
        init=args.k_means_init,
        max_iter=args.max_iter,
    )
    print(f"Number of iteratios needed by algorithm: {n_iter}")
    plot_k_means(axs[0, 1], k_means_data, assignments, rng, args.n_clusters)

    print(
        f"Image quantisation with k-means clustering (k={args.n_clusters_q}) with {args.k_means_init} initialisation"
    )

    random_img = rng.choice(os.listdir("week_7_data/imgs"))
    img = np.asarray(Image.open(f"week_7_data/imgs/{random_img}"))
    axs[1, 0].imshow(img)
    axs[1, 0].set_title("The original image", fontsize=14)

    img_seq = image_colour_quantisation(img, args.n_clusters_q, args.max_iter_q, rng)

    axs[1, 1].set_title(f"The quantised image (k={args.n_clusters_q})", fontsize=14)
    axs[1, 1].imshow(img_seq.reshape(img.shape))

    print("Loading pre-trained GloVe embeddings...")
    embeddings_path = "week_7_data/embs.txt"
    words, embeddings = load_word_embeddings(embeddings_path)

    keywords = ["car", "beach", "laptop"]
    selection = {
        k: get_nearest_neighbours(k, embeddings, words, n=10) for k in keywords
    }

    print("Compute PCA with eigendecomposition")
    pca_embeddings = pca(embeddings, method="eigen")
    plot_pca(axs[2, 0], pca_embeddings, words, selection, rng, "eigen", N_PCA)

    print("Compute PCA with SVD")
    pca_embeddings = pca(embeddings, method="svd")
    plot_pca(axs[2, 1], pca_embeddings, words, selection, rng, "svd", N_PCA)

    fig.tight_layout(pad=1)
    fig.savefig(args.file)
