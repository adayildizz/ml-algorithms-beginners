import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.spatial.distance as dt
import scipy.stats as stats

group_means = np.array([[-5.0, +0.0],
                        [+0.0, +5.0],
                        [+5.0, +0.0],
                        [+0.0, -5.0],
                        [+0.0, +0.0]])
group_covariances = np.array([[[+0.4, +0.0],
                               [+0.0, +6.0]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]],
                              [[+0.4, +0.0],
                               [+0.0, +6.0]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]]])

# read data into memory
data_set = np.genfromtxt("data_set.csv", delimiter = ",")

# get X values
X = data_set[:, [0, 1]]

# set number of clusters
K = 5

# should return initial parameter estimates
# as described in the homework description
def initialize_parameters(X, K):
    distances = np.stack([np.linalg.norm(group_means[i] - X, axis=1) for i in range(K)], axis=1)
    assigned_means = np.argmin(distances, axis=1) # N array

    priors = np.array([np.sum(assigned_means == i) / len(assigned_means) for i in range(K)])
    covariances = np.array([np.cov(X[assigned_means == i].T) for i in range(K)])
    means = np.array([X[assigned_means == i].mean(axis=0) for i in range(K)])
    return(means, covariances, priors)

means, covariances, priors = initialize_parameters(X, K)

# should return final parameter estimates of
# EM clustering algorithm
def em_clustering_algorithm(X, K, means, covariances, priors):
    N, d = X.shape
    h = np.zeros((N, K))

    for iteration in range(100):

        for k in range(K):
            pdf = stats.multivariate_normal.pdf(X, mean=means[k], cov=covariances[k])
            h[:, k] = priors[k] * pdf
        h /= h.sum(axis=1, keepdims=True)

        old_means = means.copy()

        for k in range(K):
            Nk = h[:, k].sum()
            priors[k] = Nk / N
            means[k] = (h[:, k][:, np.newaxis] * X).sum(axis=0) / Nk
            diff = X - means[k]
            covariances[k] = (h[:, k][:, np.newaxis, np.newaxis] * (
                        diff[:, :, np.newaxis] * diff[:, np.newaxis, :])).sum(axis=0) / Nk

        if np.linalg.norm(means - old_means) < 0.01:
            break
    assignments = np.argmax(h, axis=1)
    return(means, covariances, priors, assignments)

means, covariances, priors, assignments = em_clustering_algorithm(X, K, means, covariances, priors)
print(means)
print(priors)

# should draw EM clustering results as described
# in the homework description
def draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments):
    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, K))

    for k in range(K):
        cluster_points = X[assignments == k]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[k], label=f"Cluster {k + 1}", alpha=0.6)

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("EM Clustering Results")
    plt.legend()
    plt.grid()
    plt.show()
    
draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments)