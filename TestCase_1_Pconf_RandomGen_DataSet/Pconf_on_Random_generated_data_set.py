"""A simple demo to perform synthetic experiments for Pconf classification."""

from statistics import mean
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
from sklearn import svm
import statistics


def getPositivePosterior(x, mu1, mu2, cov1, cov2, positive_prior):
    """Returns the positive posterior p(y=+1|x)."""
    conditional_positive = np.exp(-0.5 * (x - mu1).T.dot(np.linalg.inv(cov1)).dot(
        x - mu1)) / np.sqrt(np.linalg.det(cov1)*(2 * np.pi)**x.shape[0])
    conditional_negative = np.exp(-0.5 * (x - mu2).T.dot(np.linalg.inv(cov2)).dot(
        x - mu2)) / np.sqrt(np.linalg.det(cov2)*(2 * np.pi)**x.shape[0])
    marginal_dist = positive_prior * conditional_positive + \
        (1 - positive_prior) * conditional_negative
    positivePosterior = conditional_positive * positive_prior / marginal_dist
    return positivePosterior


class LinearNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearNetwork, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out


def getAccuracy(x_test, y_test, model):
    """Calculates the classification accuracy."""
    predicted = model(Variable(torch.from_numpy(x_test)))
    accuracy = np.sum(torch.sign(predicted).data.numpy() ==
                      np.matrix(y_test).T) * 1. / len(y_test)
    return accuracy


def pconfClassification(num_epochs, lr, x_train_p, x_test, y_test, r):
    model = LinearNetwork(input_size=2, output_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        inputs = Variable(torch.from_numpy(x_train_p))
        confidence = Variable(torch.from_numpy(r))
        optimizer.zero_grad()
        negative_logistic = nn.LogSigmoid()
        logistic = -1. * negative_logistic(-1. * model(inputs))
        # note that \ell_L(g) - \ell_L(-g) = -g with logistic loss
        loss = torch.sum(-model(inputs)+logistic * 1. / confidence)
        loss.backward()
        optimizer.step()
    params = list(model.parameters())
    accuracy = getAccuracy(x_test=x_test, y_test=y_test, model=model)
    return params, accuracy


def naiveClassification(num_epochs, lr, x_naive, y_naive, y_test, x_test, R):
    model = LinearNetwork(input_size=2, output_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        inputs = Variable(torch.from_numpy(x_naive))
        targets = Variable(torch.from_numpy(y_naive))
        confidence = Variable(torch.from_numpy(R))
        optimizer.zero_grad()
        negative_logistic = nn.LogSigmoid()
        logistic = -1. * negative_logistic(targets * model(inputs))
        loss = torch.sum(logistic * confidence)
        loss.backward()
        optimizer.step()
    params = list(model.parameters())
    accuracy = getAccuracy(x_test=x_test, y_test=y_test, model=model)
    return params, accuracy


def osvmClassification(nu, x_train_p, x_test, y_train, y_test):
    clf = svm.OneClassSVM(nu=nu, kernel='rbf', gamma=0.1)
    clf.fit(x_train_p)
    y_pred = clf.predict(x_test)
    accuracy = np.sum(y_pred == y_test) / len(y_pred)
    return clf, accuracy


def supervisedClassification(num_epochs, lr, x_train, x_test, y_train, y_test):
    y_train_matrix = np.matrix(y_train).T.astype('float32')
    model = LinearNetwork(input_size=2, output_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        inputs = Variable(torch.from_numpy(x_train))
        targets = Variable(torch.from_numpy(y_train_matrix))
        optimizer.zero_grad()
        negative_logistic = nn.LogSigmoid()
        logistic = -1. * negative_logistic(targets * model(inputs))
        loss = torch.sum(logistic)
        loss.backward()
        optimizer.step()
    params = list(model.parameters())
    accuracy = getAccuracy(x_test=x_test, y_test=y_test, model=model)
    return params, accuracy


def generateData(mu1, mu2, cov1, cov2, n_positive, n_negative, n_positive_test, n_negative_test):
    positive_prior = n_positive/(n_positive + n_negative)
    x_train_p = np.random.multivariate_normal(mu1, cov1, n_positive)
    x_train_n = np.random.multivariate_normal(mu2, cov2, n_negative)
    x_test_p = np.random.multivariate_normal(mu1, cov1, n_positive_test)
    x_test_n = np.random.multivariate_normal(mu2, cov2, n_negative_test)
    x_naive = np.r_[x_train_p, x_train_p]
    x_naive = x_naive.astype('float32')
    y_train = np.r_[np.ones(n_positive), -np.ones(n_negative)]
    y_test = np.r_[np.ones(n_positive_test), -np.ones(n_negative_test)]
    y_naive = np.r_[np.ones(n_positive), -np.ones(n_positive)]
    y_naive = np.matrix(y_naive).T.astype('float32')
    x_train = np.r_[x_train_p, x_train_n]
    x_train = x_train.astype('float32')
    x_test = np.r_[x_test_p, x_test_n]
    x_test = x_test.astype('float32')
    x_train_p = x_train_p.astype('float32')
    # calculating the exact positive-confidence values: r
    r = np.zeros(n_positive)
    for i in range(n_positive):
        x = x_train_p[i, :]
        r[i] = getPositivePosterior(x, mu1, mu2, cov1, cov2, positive_prior)
    R = np.r_[r, 1-r]
    r = np.matrix(r).T
    r = r.astype('float32')
    R = np.matrix(R).T.astype('float32')

    return r, R, x_naive, x_train, x_train_p, x_test, y_naive, y_train, y_test


n_positive, n_negative = 500, 500
n_positive_test, n_negative_test = 1000, 1000
mu1 = [0, 0]
mu2 = [[-2, 5], [0, 4], [0, 8], [0, 4]]
cov1_candidates = [[[7, -6], [-6, 7]], [[5, 3], [3, 5]],
                   [[7, -6], [-6, 7]], [[4, 0], [0, 4]]]
cov2_candidates = [[[2, 0], [0, 2]], [[5, -3], [-3, 5]],
                   [[7, 6], [6, 7]], [[1, 0], [0, 1]]]

lr = 0.001
num_epochs = 5000

# import pdb
# pdb.set_trace()


def cal_average_pconf_classification():

    test_mean_pconf = []
    test_sd_pconf = []
    test_list = ["A", "B", "C", "D"]

    for i in range(0, 4):
        pconf_accuracy = []
        print("Calculating accuracies for pconf Test {0}".format(test_list[i]))

        for j in range(0, 20):

            temp_mu2 = mu2[i]
            cov1, cov2 = cov1_candidates[i], cov2_candidates[i]
            r, R, x_naive, x_train, x_train_p, x_test, y_naive, y_train, y_test = generateData(
                mu1, temp_mu2, cov1, cov2, n_positive, n_negative, n_positive_test, n_negative_test)

            params_pconf, pconf_acc = pconfClassification(
                num_epochs, lr, x_train_p, x_test, y_test, r)

            pconf_accuracy.append(round(pconf_acc * 100, 2))

            print("Test {0} {1} {2}".format(test_list[i], j+1, pconf_accuracy))

        mean_accuracy = mean(pconf_accuracy)
        sd_accuracy = statistics.stdev(pconf_accuracy)
        test_mean_pconf.append(mean_accuracy)
        test_sd_pconf.append(sd_accuracy)
        print("Accuracies calculated for pconf Test {0}".format(test_list[i]))

    return test_mean_pconf, test_sd_pconf


def cal_average_osvmClassification():

    test_mean_osvm = []
    test_sd_osvm = []
    test_list = ["A", "B", "C", "D"]
    nu = 0.05

    for i in range(0, 4):
        osvm_accuracy = []
        print("Calculating accuracies for osvm Test {0}".format(test_list[i]))

        for j in range(0, 20):

            temp_mu2 = mu2[i]
            cov1, cov2 = cov1_candidates[i], cov2_candidates[i]
            r, R, x_naive, x_train, x_train_p, x_test, y_naive, y_train, y_test = generateData(
                mu1, temp_mu2, cov1, cov2, n_positive, n_negative, n_positive_test, n_negative_test)

            params_, osvm_acc = osvmClassification(
                nu, x_train_p, x_test, y_train, y_test)

            osvm_accuracy.append(round(osvm_acc * 100, 2))

            print("Test {0} {1} {2}".format(test_list[i], j+1, osvm_accuracy))

        accuracy = mean(osvm_accuracy)
        sd_accuracy = statistics.stdev(osvm_accuracy)
        test_mean_osvm.append(accuracy)
        test_sd_osvm.append(sd_accuracy)
        print("Accuracies calculated for osvm Test {0}".format(test_list[i]))

    return test_mean_osvm, test_sd_osvm


def cal_average_supervisedClassification():

    test_mean_super = []
    test_sd_super = []
    test_list = ["A", "B", "C", "D"]

    for i in range(0, 4):
        super_accuracy = []
        print("Calculating accuracies for super Test {0}".format(test_list[i]))

        for j in range(0, 20):

            temp_mu2 = mu2[i]
            cov1, cov2 = cov1_candidates[i], cov2_candidates[i]
            r, R, x_naive, x_train, x_train_p, x_test, y_naive, y_train, y_test = generateData(
                mu1, temp_mu2, cov1, cov2, n_positive, n_negative, n_positive_test, n_negative_test)

            params_, super_acc = supervisedClassification(
                num_epochs, lr, x_train, x_test, y_train, y_test)

            super_accuracy.append(round(super_acc * 100, 2))

            print("Test {0} {1} {2}".format(test_list[i], j+1, super_accuracy))

        accuracy = mean(super_accuracy)
        sd_accuracy = statistics.stdev(super_accuracy)
        test_mean_super.append(accuracy)
        test_sd_super.append(sd_accuracy)
        print("Accuracies calculated for super Test {0}".format(test_list[i]))

        print("\n")

    return test_mean_super, test_sd_super


def cal_average_naiveClassification():

    test_mean_naive = []
    test_sd_naive = []
    test_list = ["A", "B", "C", "D"]

    for i in range(0, 4):
        naive_accuracy = []
        print("Calculating accuracies for naive Test {0}".format(test_list[i]))

        for j in range(0, 20):

            temp_mu2 = mu2[i]
            cov1, cov2 = cov1_candidates[i], cov2_candidates[i]
            r, R, x_naive, x_train, x_train_p, x_test, y_naive, y_train, y_test = generateData(
                mu1, temp_mu2, cov1, cov2, n_positive, n_negative, n_positive_test, n_negative_test)

            params_, naive_acc = naiveClassification(num_epochs, lr, x_naive, y_naive, y_test, x_test, R)

            naive_accuracy.append(round(naive_acc * 100, 2))

            print("Test {0} {1} {2}".format(test_list[i], j+1, naive_accuracy))

        accuracy = mean(naive_accuracy)
        sd_accuracy = statistics.stdev(naive_accuracy)
        test_mean_naive.append(accuracy)
        test_sd_naive.append(sd_accuracy)
        print("Accuracies calculated for naive Test {0}".format(test_list[i]))

        print("\n")

    return test_mean_naive, test_sd_naive


if __name__ == "__main__":

    print("\n" + ("*" * 100))
    pconf_mean_accuracy, pconf_sd_accuracy = cal_average_pconf_classification()
    print(("*" * 100))

    print("\n" + ("*" * 100))
    naive_mean_accuracy, naive_sd_accuracy = cal_average_naiveClassification()
    print(("*" * 100))

    print("\n" + ("*" * 100))
    osvm_mean_accuracy, osvm_sd_accuracy = cal_average_osvmClassification()
    print(("*" * 100))

    print("\n" + ("*" * 100))
    super_mean_accuracy, super_sd_accuracy = cal_average_supervisedClassification()
    print(("*" * 100))

    print("\n")
    print("Mean and Standard Deviation Accuracy for Test A pconf after 20 reps {0} {1}".format(pconf_mean_accuracy[0], pconf_sd_accuracy[0]))
    print("Mean and Standard Deviation Accuracy for Test B pconf after 20 reps {0} {1}".format(pconf_mean_accuracy[1], pconf_sd_accuracy[1]))
    print("Mean and Standard Deviation Accuracy for Test C pconf after 20 reps {0} {1}".format(pconf_mean_accuracy[2], pconf_sd_accuracy[2]))
    print("Mean and Standard Deviation Accuracy for Test D pconf after 20 reps {0} {1}".format(pconf_mean_accuracy[3], pconf_sd_accuracy[3]))

    print("\n")
    print("Mean and Standard Deviation Accuracy for Test A naive after 20 reps {0} {1}".format(naive_mean_accuracy[0], naive_sd_accuracy[0]))
    print("Mean and Standard Deviation Accuracy for Test B naive after 20 reps {0} {1}".format(naive_mean_accuracy[1], naive_sd_accuracy[1]))
    print("Mean and Standard Deviation Accuracy for Test C naive after 20 reps {0} {1}".format(naive_mean_accuracy[2], naive_sd_accuracy[2]))
    print("Mean and Standard Deviation Accuracy for Test D naive after 20 reps {0} {1}".format(naive_mean_accuracy[3], naive_sd_accuracy[3]))

    print("\n")
    print("Mean and Standard Deviation Accuracy for Test A osvm after 20 reps {0} {1}".format(osvm_mean_accuracy[0], osvm_sd_accuracy[0]))
    print("Mean and Standard Deviation Accuracy for Test B osvm after 20 reps {0} {1}".format(osvm_mean_accuracy[1], osvm_sd_accuracy[1]))
    print("Mean and Standard Deviation Accuracy for Test C osvm after 20 reps {0} {1}".format(osvm_mean_accuracy[2], osvm_sd_accuracy[2]))
    print("Mean and Standard Deviation Accuracy for Test D osvm after 20 reps {0} {1}".format(osvm_mean_accuracy[3], osvm_sd_accuracy[3]))

    print("\n")
    print("Mean and Standard Deviation Accuracy for Test A super after 20 reps {0} {1}".format(super_mean_accuracy[0], super_sd_accuracy[0]))
    print("Mean and Standard Deviation Accuracy for Test B super after 20 reps {0} {1}".format(super_mean_accuracy[1], super_sd_accuracy[1]))
    print("Mean and Standard Deviation Accuracy for Test C super after 20 reps {0} {1}".format(super_mean_accuracy[2], super_sd_accuracy[2]))
    print("Mean and Standard Deviation Accuracy for Test D super after 20 reps {0} {1}".format(super_mean_accuracy[3], super_sd_accuracy[3]))
