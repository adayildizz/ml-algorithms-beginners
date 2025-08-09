import math
import matplotlib.pyplot as plt
import numpy as np
import os 
os.chdir("D:/ml-algorithms-beginners/non-parametric_methods")

# read data into memory
data_set_train = np.genfromtxt("data_set_train.csv", delimiter = ",", skip_header = 1)
data_set_test = np.genfromtxt("data_set_test.csv", delimiter = ",", skip_header = 1)

# get X and y values
X_train = data_set_train[:, 0:2]
y_train = data_set_train[:, 2]
X_test = data_set_test[:, 0:2]
y_test = data_set_test[:, 2]

minimum_value = -2.0
maximum_value = +2.0

def plot_figure(y, y_hat):
    fig = plt.figure(figsize = (4, 4))
    plt.axline([-12, -12], [52, 52], color = "r", linestyle = "--")
    plt.plot(y, y_hat, "k.")
    plt.xlabel("True value ($y$)")
    plt.ylabel("Predicted value ($\widehat{y}$)")
    plt.xlim([-12, 52])
    plt.ylim([-12, 52])
    plt.show()
    return(fig)

# assuming that there are N query data points
# should return a numpy array with shape (N,)
def regressogram(X_query, X_train, y_train, x1_left_borders, x1_right_borders, x2_left_borders, x2_right_borders):
    num_bins = len(x1_left_borders)
    N = len(X_query)
    query_bins = np.zeros((N, num_bins))

    for b in range(num_bins):
        x1_in_bin = (x1_left_borders[b] < X_query[:, 0]) & (X_query[:, 0] <= x1_right_borders[b])
        x2_in_bin = (x2_left_borders[b] < X_query[:, 1]) & (X_query[:, 1] <= x2_right_borders[b])
        
        query_bins[:, b] = x1_in_bin & x2_in_bin

    y_hat = np.zeros(N)
    for i in range(N):
        bin_idx = np.argmax(query_bins[i])  
        
        in_bin = (
            (x1_left_borders[bin_idx] < X_train[:, 0]) & (X_train[:, 0] <= x1_right_borders[bin_idx]) &
            (x2_left_borders[bin_idx] < X_train[:, 1]) & (X_train[:, 1] <= x2_right_borders[bin_idx])
        )
        
        if np.sum(in_bin) > 0:
            y_hat[i] = np.mean(y_train[in_bin])
        else:
            y_hat[i] = 0  

    return(y_hat)
    
bin_width = 0.50
left_borders = np.arange(start = minimum_value, stop = maximum_value, step = bin_width)
right_borders = np.arange(start = minimum_value + bin_width, stop = maximum_value + bin_width, step = bin_width)

x1_left_borders = np.meshgrid(left_borders, left_borders)[0].flatten()
x1_right_borders = np.meshgrid(right_borders, right_borders)[0].flatten()
x2_left_borders = np.meshgrid(left_borders, left_borders)[1].flatten()
x2_right_borders = np.meshgrid(right_borders, right_borders)[1].flatten()

y_train_hat = regressogram(X_train, X_train, y_train, x1_left_borders, x1_right_borders, x2_left_borders, x2_right_borders)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("Regressogram => RMSE on training set is {} when h is {}".format(rmse, bin_width))

fig = plot_figure(y_train, y_train_hat)
fig.savefig("regressogram_training.pdf", bbox_inches = "tight")

y_test_hat = regressogram(X_test, X_train, y_train, x1_left_borders, x1_right_borders, x2_left_borders, x2_right_borders)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Regressogram => RMSE on test set is {} when h is {}".format(rmse, bin_width))

fig = plot_figure(y_test, y_test_hat)
fig.savefig("regressogram_test.pdf", bbox_inches = "tight")


# assuming that there are N query data points
# should return a numpy array with shape (N,)
def running_mean_smoother(X_query, X_train, y_train, bin_width):
    N = len(X_query)
    y_hat = np.zeros(N)

    for i in range(N):

        mask1 = ((X_query[i, 0]-(bin_width/2))< X_train[:,0]) & ((X_query[i,0]+(bin_width/2)) >= X_train[:,0])
        mask2 = ((X_query[i, 1]-(bin_width/2))< X_train[:, 1]) & ((X_query[i,1]+(bin_width/2)) >= X_train[:, 1])
        
        if np.sum((mask1&mask2)) > 0:
            y_hat[i] = np.mean(y_train[(mask1 & mask2)])
        else:
            y_hat[i] = 0

    return(y_hat)

bin_width = 0.50

y_train_hat = running_mean_smoother(X_train, X_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("Running Mean Smoother => RMSE on training set is {} when h is {}".format(rmse, bin_width))

fig = plot_figure(y_train, y_train_hat)
fig.savefig("running_mean_smoother_training.pdf", bbox_inches = "tight")

y_test_hat = running_mean_smoother(X_test, X_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Running Mean Smoother => RMSE on test set is {} when h is {}".format(rmse, bin_width))

fig = plot_figure(y_test, y_test_hat)
fig.savefig("running_mean_smoother_test.pdf", bbox_inches = "tight")

# assuming that there are N query data points
# should return a numpy array with shape (N,)
def kernel_smoother(X_query, X_train, y_train, bin_width):
 
    N = len(X_query)
    y_hat = np.zeros(N)
    u = np.zeros(N)
    for i in range(N):
        diff = X_query[i] - X_train           
        u = np.linalg.norm(diff, axis=1) / bin_width  
        kernels = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)

        if np.sum(kernels) == 0:
            y_hat[i] = 0
        else:
            y_hat[i] = np.sum(kernels * y_train) / np.sum(kernels)

    return(y_hat)

bin_width = 0.08

y_train_hat = kernel_smoother(X_train, X_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("Kernel Smoother => RMSE on training set is {} when h is {}".format(rmse, bin_width))

fig = plot_figure(y_train, y_train_hat)
fig.savefig("kernel_smoother_training.pdf", bbox_inches = "tight")

y_test_hat = kernel_smoother(X_test, X_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Kernel Smoother => RMSE on test set is {} when h is {}".format(rmse, bin_width))

fig = plot_figure(y_test, y_test_hat)
fig.savefig("kernel_smoother_test.pdf", bbox_inches = "tight")
