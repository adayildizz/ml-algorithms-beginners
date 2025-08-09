import numpy as np
import pandas as pd
import os 
os.chdir("D:/ml-algorithms-beginners/naive_bayesian_classifier")

X_train = np.genfromtxt("20newsgroup_words_train.csv", delimiter = ",", dtype = int)
y_train = np.genfromtxt("20newsgroup_labels_train.csv", delimiter = ",", dtype = int)
X_test = np.genfromtxt("20newsgroup_words_test.csv", delimiter = ",", dtype = int)
y_test = np.genfromtxt("20newsgroup_labels_test.csv", delimiter = ",", dtype = int)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


def estimate_prior_probabilities(y):
    K = 20
    class_priors = [np.mean(y == (c+1)) for c in range(K)]
    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)
print(class_priors)

# assuming that there are K classes and D features
# should return a numpy array with shape (K, D)
def estimate_success_probabilities(X, y):
    K = 20
    D = 2000 
    alpha = 0.2 
    # WARNINGS
    # if no np.array, P becomes a simple list, the reason that K and D are accepted as dimensions is this. (afaik)
    # Note that alpha is added outside the sum operation in nominator
    # we are summing through axis=0, since we are summing the data points if they are in class c. 
    P = np.array([(np.sum(X[y == (c+1)], axis=0) + alpha) / (np.sum(y == (c+1)) + alpha*D) for c in range(K)])
    # cleaner ways can be found later.
    return(P)

P = estimate_success_probabilities(X_train, y_train)
print(P)


# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, P, class_priors):
    N = X.shape[0]
    K = 20
    D = 2000
    score_values = np.zeros((N, K))
    for c in range(K):
        score_values[:, c] = X @ np.log(P[c, :]) + (1 - X) @ np.log(1 - P[c, :]) + np.log(class_priors[c])

    return(score_values)

scores_train = calculate_score_values(X_train, P, class_priors)
print(scores_train)

scores_test = calculate_score_values(X_test, P, class_priors)
print(scores_test)

# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # need to define confusion matrix:
    # each row is the actual class
    # each column is the number of prediction jth column while actual was ith row
    # some search made for np.bincount : counts the occurance of each element, minlength is needed to be able to add zero occurences to the end.
    K = 20
    y_predicted = np.argmax(scores, axis=1)
    confusion_matrix = np.zeros((K, K), dtype=int)
    for c in range(K):
        confusion_matrix[c, :] = np.bincount(y_predicted[y_truth == c+1], minlength=K)
    print(confusion_matrix)
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print("Training accuracy is {:.2f}%.".format(100 * np.sum(np.diag(confusion_train)) / np.sum(confusion_train)))

confusion_test = calculate_confusion_matrix(y_test, scores_test)
print("Test accuracy is {:.2f}%.".format(100 * np.sum(np.diag(confusion_test)) / np.sum(confusion_test)))
