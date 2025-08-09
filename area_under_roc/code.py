import matplotlib.pyplot as plt
import numpy as np
import os 
os.chdir("D:/ml-algorithms-beginners/area_under_roc")

true_labels = np.genfromtxt("true_labels.csv", delimiter = ",", dtype = "int")
predicted_probabilities1 = np.genfromtxt("predicted_probabilities1.csv", delimiter = ",")
predicted_probabilities2 = np.genfromtxt("predicted_probabilities2.csv", delimiter = ",")

# STEP 3
# given the predicted probabilities of size (N,),
# it should return the calculated thresholds of size (N + 1,)
def calculate_threholds(predicted_probabilities):
    # your implementation starts below
    sorted_probabilities = np.sort(predicted_probabilities)
    thresholds = []
    thresholds.append(0.0)
    for p in sorted_probabilities:
        thresholds.append(p)
    thresholds.append(1.0)
    thresholds = np.array(thresholds)
    # your implementation ends above
    return thresholds

thresholds1 = calculate_threholds(predicted_probabilities1)
print(thresholds1)

thresholds2 = calculate_threholds(predicted_probabilities2)
print(thresholds2)

# STEP 4
# given the true labels of size (N,), the predicted probabilities of size (N,) and
# the thresholds of size (N + 1,), it should return the FP and TP rates of size (N + 1,)
def calculate_fp_and_tp_rates(true_labels, predicted_probabilities, thresholds):
    # your implementation starts below
    fp_rates = []
    tp_rates = []

    P = np.sum(true_labels == 1)
    N = np.sum(true_labels == -1)

    for threshold in thresholds:
        predictions = (predicted_probabilities >= threshold).astype(int)

        TP = np.sum((predictions == 1) & (true_labels == 1))
        FP = np.sum((predictions == 1) & (true_labels == -1))

        tp_rate = TP / P if P > 0 else 0
        fp_rate = FP / N if N > 0 else 0

        tp_rates.append(tp_rate)
        fp_rates.append(fp_rate)

    # your implementation ends above
    return fp_rates, tp_rates

fp_rates1, tp_rates1 = calculate_fp_and_tp_rates(true_labels, predicted_probabilities1, thresholds1)
print(fp_rates1[495:505])
print(tp_rates1[495:505])

fp_rates2, tp_rates2 = calculate_fp_and_tp_rates(true_labels, predicted_probabilities2, thresholds2)
print(fp_rates2[495:505])
print(tp_rates2[495:505])

fig = plt.figure(figsize = (5, 5))
plt.plot(fp_rates1, tp_rates1, label = "Classifier 1")
plt.plot(fp_rates2, tp_rates2, label = "Classifier 2")
plt.xlabel("FP Rate")
plt.ylabel("TP Rate")
plt.legend()
plt.show()
fig.savefig("hw06_roc_curves.pdf", bbox_inches = "tight")

# STEP 5
# given the FP and TP rates of size (N + 1,),
# it should return the area under the ROC curve
def calculate_auroc(fp_rates, tp_rates):
    # your implementation starts below
    auroc = 0.0
    for i in range(1, len(fp_rates)):
        auroc += (fp_rates[i] - fp_rates[i-1]) * (tp_rates[i] + tp_rates[i-1]) / 2
    # your implementation ends above
    return auroc

auroc1 = calculate_auroc(fp_rates1, tp_rates1)
print("The area under the ROC curve for Algorithm 1 is {}.".format(auroc1))
auroc2 = calculate_auroc(fp_rates2, tp_rates2)
print("The area under the ROC curve for Algorithm 2 is {}.".format(auroc2))
