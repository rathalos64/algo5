#!/usr/bin/env python

import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

from functools import reduce
    
from mpl_toolkits.mplot3d import Axes3D

from sklearn.datasets.samples_generator import make_blobs

# TODO:
# * compare multivariate pdf with scipy.stats's implementation
#   The probability densities are the same although the scaling differs.
#   For my implementation, based on some mu and cov, the max p differes up to
#   0.06 to numpy's impelementation. This affects the log-likelihood
#   Me (-5032.44494116788) vs Scipy (-3384.526508165725).
#   [NOTE]: This only concerns the scaling factor. The kernel function was implemented
#   correctly.

def main():
    mean = np.array([0, 0])
    cov = np.array([
        [2, 1.0], 
        [1.0, 2]
    ])

    # samples, _ = make_blobs(
    #     n_samples=500,
    #     n_features=2,
    #     centers=mu,
    #     cluster_std=cov,
    # )

    samples = np.random.multivariate_normal(mean=mean, cov=cov, size=(1000))
    densities = np.array([multivariate_normal_pdf(sample, mean, cov) for sample in samples])

    print(_loglikelihood(samples, mean, cov))
    # print(_loglikelihood_numpy(samples, mean, cov))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(samples.T[0], samples.T[1], densities, linestyle="None", c="g", marker=".", depthshade=True)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    densies = multivariate_normal.pdf(samples, mean=mean, cov=cov)
    ax2.scatter(samples.T[0], samples.T[1], densies, linestyle="None", c="r", marker=".", depthshade=True)

    plt.show()

# multivariate_normal_pdf calculates the density of a given observation x based
# on a multivariate normal distribution parametrised by the mean and the covariance matrix 
def multivariate_normal_pdf(x, mean, cov):
    D = len(x)

    # for python: * is element-wise multiplication, not matrix multiplication
    # use dot instead
    mahalanobis = (x - mean).T.dot(np.linalg.inv(cov)).dot((x - mean))
    normalizing = math.pow(2 * math.pi, D / 2) * math.pow(np.linalg.det(cov), 2)
    
    return math.exp(-0.5 * mahalanobis) / normalizing

# _loglikelihood calculate the logarithmic likelihood of data X given the parameters
# of a multivariate normal distribution (mean and covariance matrix)
def _loglikelihood(X, mean, cov):
    return reduce(lambda acc, x: acc + math.log(multivariate_normal_pdf(x, mean, cov)), X, 0)

def _loglikelihood_numpy(X, mean, cov):
    Y = multivariate_normal.pdf(X, mean=mean, cov=cov)
    return reduce(lambda acc, x: acc + math.log(x), Y, 0)

if __name__ == "__main__":
    main()
