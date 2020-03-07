#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import matplotlib
import platform

if platform.system() == "Darwin":
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import entropy
from input_fn import *
import numpy as np
import argparse

NUM_PRINT_TERMS = 5


def _jensen_shannon(_P, _Q):
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


def dxtp_parameter_estimation(kw, kt, ks, ztot, alpha, beta):
    num_topics = kw.shape[0]
    num_vocabs = kw.shape[1]
    num_tools = kt.shape[1]
    num_datasets = ks.shape[1]

    est_kw = np.zeros((num_topics, num_vocabs))
    est_kt = np.zeros((num_topics, num_tools))
    est_ks = np.zeros((num_topics, num_datasets))

    for k in xrange(num_topics):
        est_kw[k, :] = (1.0 * (kw[k, :] + beta)) / (ztot[k] + num_vocabs * beta)
        est_kt[k, :] = (1.0 * (kt[k, :] + alpha)) / (np.sum(kt[k, :]) + num_tools * alpha)
        est_ks[k, :] = (1.0 * (ks[k, :] + alpha)) / (np.sum(ks[k, :]) + num_datasets * alpha)

    return est_kw, est_kt, est_ks


def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape

    # replace euclidean distance with MMD
    # sum_X = np.sum(np.square(X), 1)
    # D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                p = X[i, :]
                q = X[j, :]
                max_pq = np.max(np.vstack((X[i, :], X[j, :])), axis=0)       # in order to symmetric distance
                D[i, j] = np.sum(np.square(p - q) / max_pq)
                # D[i, j] = (np.sum(np.square(p - q) / p) + np.sum(np.square(p - q) / q)) / 2

    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0, max_iter=1000):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    # X = pca(X, initial_dims).real
    (n, d) = X.shape
    print X.shape
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):
        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        C = np.sum(P * np.log(P / Q))
        if (iter + 1) % 10 == 0:
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y, C


def get_topic_distribution(domain, folder):
    # get inputs data
    inputs = input_fn('demo', domain)
    vocabs = inputs['vocab']
    tools = inputs['tools']
    datasets = inputs['datasets']

    # load model file
    folder_name = MODELS_FOLDER + folder + '/'
    kw = np.loadtxt(folder_name + 'kw.dat')
    kt = np.loadtxt(folder_name + 'kt.dat')
    ks = np.loadtxt(folder_name + 'ks.dat')
    ztot = np.loadtxt(folder_name + 'ztot.dat')

    with open(MODELS_FOLDER + folder + '/settings.json') as fp:
        settings = json.load(fp)  # docs with bag of words
        alpha = settings['alpha']
        beta = settings['beta']

    est_kw, _, _ = dxtp_parameter_estimation(kw, kt, ks, ztot, alpha, beta)

    return est_kw, vocabs


def get_data_source(folder_name):
    with open(MODELS_FOLDER + folder_name + '/settings.json') as fp:
        settings = json.load(fp)  # docs with bag of words
        data_source = settings['data_source']

        return data_source


def single_domain(folder, num_iterations):
    domain = get_data_source(folder)

    est_kw, vocabs = get_topic_distribution(domain, folder)
    num_topics = est_kw.shape[0]

    log_likelihood = np.sum(np.log(est_kw), axis=1, keepdims=0)
    log_likelihood = abs(log_likelihood) - np.min(abs(log_likelihood)) + 1

    topic_summaries = [''] * 100
    for k in xrange(num_topics):
        kw_idx = np.argsort(est_kw[k, :])[::-1][:NUM_PRINT_TERMS]
        for x in xrange(NUM_PRINT_TERMS):
            topic_summaries[k] += vocabs[kw_idx[x]]
            if x != NUM_PRINT_TERMS - 1:
                topic_summaries[k] += '\n'

    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    Y, _ = tsne(est_kw, 2, 50, 10.0, num_iterations)

    # plot
    volume = (log_likelihood / 1000) * 10 + 1

    fig, ax = plt.subplots()
    ax.scatter(Y[:, 0], Y[:, 1], s=volume, c='g', alpha=0.5)

    # add text
    for i in xrange(num_topics):
        plt.text(Y[i, 0], Y[i, 1], topic_summaries[i], horizontalalignment='center',
                 verticalalignment='center', fontsize=3)

    ax.set_axis_off()
    ax.grid(False)
    fig.tight_layout()
    plt.show()

    title = "tsne_embedding_single_domain_" + domain + '.pdf'
    with PdfPages(MODELS_FOLDER + title) as pdf:
        pdf.savefig(fig)


def cross_domain(folder1, folder2, num_iterations):
    domain1 = get_data_source(folder1)
    domain2 = get_data_source(folder2)

    if domain1 == 'bio':
        est_kw_bio, vocabs = get_topic_distribution('bio', folder1)
        est_kw_neuro, _ = get_topic_distribution('neuro', folder2)
    else:
        est_kw_bio, vocabs = get_topic_distribution('bio', folder2)
        est_kw_neuro, _ = get_topic_distribution('neuro', folder1)

    est_kw = np.concatenate((est_kw_bio, est_kw_neuro))

    # # normalize to 1
    # for i in range(est_kw.shape[0]):
    #     est_kw[i, :] = est_kw[i, :] / np.sum(est_kw[i, :])

    num_topics = est_kw.shape[0]

    log_likelihood = np.sum(np.log(est_kw), axis=1, keepdims=0)
    log_likelihood = abs(log_likelihood) - np.min(abs(log_likelihood)) + 1

    topic_summaries = [''] * num_topics
    for k in xrange(num_topics):
        kw_idx = np.argsort(est_kw[k, :])[::-1][:NUM_PRINT_TERMS]
        for x in xrange(NUM_PRINT_TERMS):
            topic_summaries[k] += vocabs[kw_idx[x]]
            if x != NUM_PRINT_TERMS - 1:
                topic_summaries[k] += '\n'

    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    Y, C = tsne(est_kw, 2, 50, 10.0, num_iterations)

    # plot
    volume = (log_likelihood / 1000) * 8 + 1

    Y1 = Y[0: est_kw_bio.shape[0], :]
    volume1 = volume[0: est_kw_bio.shape[0]]

    Y2 = Y[est_kw_bio.shape[0]:, :]
    volume2 = volume[est_kw_bio.shape[0]:]
    fig, ax = plt.subplots()

    ax.scatter(Y1[:, 0], Y1[:, 1], s=volume1 / 4, c='1.0', edgecolors="0.6")
    ax.scatter(Y2[:, 0], Y2[:, 1], s=volume2 / 4, c='0.6', edgecolors="0.6")

    # add text
    for i in xrange(num_topics):
        plt.text(Y[i, 0], Y[i, 1], topic_summaries[i], horizontalalignment='center',
                 verticalalignment='center', fontsize=3)

    ax.set_axis_off()
    ax.grid(False)
    fig.tight_layout()
    plt.show()

    pdf_name = "tsne_embedding_cross_domain" + ".pdf"
    with PdfPages(MODELS_FOLDER + pdf_name) as pdf:
        pdf.savefig(fig)


def parse_args():
    parser = argparse.ArgumentParser(description='Description the Command Line of DSTM Model tSNE demonstration')
    parser.add_argument('--type', help='Specify type of demonstration (single domain, cross domain)', type=str, choices=['single', 'cross'])
    parser.add_argument('--model_folder1', help='Specify the model folder name.', type=str)
    parser.add_argument('--model_folder2', help='Specify the model folder name, when you choose cross domain '
                                                'demonstration', type=str)
    parser.add_argument('--num_iterations', help='Number of iterations (default = %(default)s)', type=int,
                        default=1000, choices=range(0, 10001), metavar='(0, ..., 10001)')

    args = parser.parse_args()

    if args.type is None:
        parser.error('Please provide type of demonstration (single domain, cross domain)')

    if args.model_folder1 is None:
        parser.error('Please provide the model folder name for the first model')

    if args.type == 'cross' and args.model_folder2 is None:
        parser.error('Please provide the model folder name for the second model')

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.type == 'single':
        single_domain(args.model_folder1, args.num_iterations)
    elif args.type == 'cross':
        cross_domain(args.model_folder1, args.model_folder2, args.num_iterations)
