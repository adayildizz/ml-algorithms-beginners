import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 
os.chdir("D:/ml-algorithms-beginners/discrimination_by_regression")


X = np.genfromtxt("fashionmnist_data_points.csv", delimiter = ",") / 255
y = np.genfromtxt("fashionmnist_class_labels.csv", delimiter = ",").astype(int)



i1 = np.hstack((np.reshape(X[np.where(y == 1)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 2)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 3)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 4)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 5)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 6)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 7)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 8)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 9)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 10)[0][0:5], :], (28 * 5, 28))))

fig = plt.figure(figsize = (10, 5))
plt.axis("off")
plt.imshow(i1, cmap = "gray")
plt.show()
fig.savefig("hw02_images.pdf", bbox_inches = "tight")



# first 60000 data points will be included to train
# remaining 10000 data points will be included to test
# should return X_train, y_train, X_test, and y_test
def train_test_split(X, y):
    X_train = X[0:60000, :]
    X_test = X[60000:, :]

    y_train = y[0:60000]
    y_test = y[60000:]

    return(X_train, y_train, X_test, y_test)

X_train, y_train, X_test, y_test = train_test_split(X, y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def sigmoid(X, W, w0):
    z = np.matmul(X, W) + w0
    scores = 1 / (1 + np.exp(-z))
    return(scores)

# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def one_hot_encoding(y):
    N = y.shape[0] #size of the train data 
    K = 10
    Y = np.zeros((N, K))
    Y[np.arange(N), (y-1)] = 1
    return(Y)




np.random.seed(421)
D = X_train.shape[1]
K = np.max(y_train)
Y_train = one_hot_encoding(y_train)
W_initial = np.random.uniform(low = -0.001, high = 0.001, size = (D, K))
w0_initial = np.random.uniform(low = -0.001, high = 0.001, size = (1, K))

# assuming that there are D features and K classes
# should return a numpy array with shape (D, K)
def gradient_W(X, Y_truth, Y_predicted):
    delta = (Y_predicted - Y_truth) * Y_predicted * (1 - Y_predicted)
    gradient = np.dot(X.T, delta)
    return(gradient)

# assuming that there are K classes
# should return a numpy array with shape (1, K)
def gradient_w0(Y_truth, Y_predicted):
    delta = (Y_predicted - Y_truth) * Y_predicted * (1 - Y_predicted)
    gradient = np.sum(delta, axis=0, keepdims=True)
    return(gradient)


# assuming that there are N data points and K classes
# should return three numpy arrays with shapes (D, K), (1, K), and (500,)
def discrimination_by_regression(X_train, Y_train,
                                 W_initial, w0_initial):
    eta = 0.15 / X_train.shape[0]
    iteration_count = 500

    W = W_initial
    w0 = w0_initial
        

    objective_values =[]
    for i in range(iteration_count):
        y_predicted = sigmoid(X_train, W, w0)

        error = 0.5 * (Y_train - y_predicted)**2
        objective_values.append(np.sum(error))


        delta_W = gradient_W(X_train, Y_train, y_predicted)
        delta_w0 = gradient_w0(Y_train, y_predicted)

        W = W - eta*delta_W
        w0 = w0 - eta*delta_w0

    return(W, w0, objective_values)

W, w0, objective_values = discrimination_by_regression(X_train, Y_train,
                                                       W_initial, w0_initial)
print(W)
print(w0)
print(objective_values[0:10])


fig = plt.figure(figsize = (10, 6))
plt.plot(range(1, len(objective_values) + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()
fig.savefig("hw02_iterations.pdf", bbox_inches = "tight")


# assuming that there are N data points
# should return a numpy array with shape (N,)
def calculate_predicted_class_labels(X, W, w0):
    scores = sigmoid(X, W, w0) # returns shape (N, K)
    y_predicted = np.argmax(scores, axis=1) + 1 # snce we are using 1 indexing
    return(y_predicted)

y_hat_train = calculate_predicted_class_labels(X_train, W, w0)
print(y_hat_train)

y_hat_test = calculate_predicted_class_labels(X_test, W, w0)
print(y_hat_test)

# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, y_predicted):
    K = 10
    confusion_matrix = pd.crosstab(y_predicted.T, y_truth.T, 
                               rownames = ["y_pred"], 
                               colnames = ["y_truth"])
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, y_hat_train)
print(confusion_train)

confusion_test = calculate_confusion_matrix(y_test, y_hat_test)
print(confusion_test)
