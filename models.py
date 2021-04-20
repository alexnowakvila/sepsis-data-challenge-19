import numpy as np, os, sys
import pdb
import time
from pandas import DataFrame
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# from get_sepsis_score import get_sepsis_score as get_sepsis_score_baseline

""" Model class """


class IndependentModel(object):
    def __init__(self, classifier, classifier_name, k=0):
        self.clf = classifier
        self.name = classifier_name
        self.scaler = StandardScaler()
        self.k = k
        # covariates to select
        self.covariates = None
        self.means = None

    def preprocess_data_test(self, X):
        X = np.nan_to_num(X)
        # X = X[:, self.covariates]
        X = self.scaler.transform(X)
        # pdb.set_trace()
        return X

    def fit(self, X_train, y_train):
        # convert to matrix
        data_x = np.concatenate(X_train, axis=0)  # concat data
        data_y = np.concatenate(y_train, axis=0)  # concat data
        # deal with NaNs
        data_x = np.nan_to_num(data_x)
        # self.covariates = ~np.isnan(data_x[0])
        # data_x = data_x[:, self.covariates]
        data_x = self.scaler.fit_transform(data_x)
        # TODO: concatenate features according to k
        # train model
        self.clf.fit(data_x, data_y)

    def get_sepsis_score(self, data):
        x = data[-self.k-1:]
        # pdb.set_trace()
        label = self.clf.predict(x)
        score = label
        if hasattr(self.clf, "predict_proba"):
            score = self.clf.predict_proba(x)[0][1]
        # print(time.time() - start)
        # pdb.set_trace()
        return score, label

    def get_sepsis_score_baseline(self, data):
        x_mean = np.array([
            83.8996, 97.0520,  36.8055,  126.2240, 86.2907,
            66.2070, 18.7280,  33.7373,  -3.1923,  22.5352,
            0.4597,  7.3889,   39.5049,  96.8883,  103.4265,
            22.4952, 87.5214,  7.7210,   106.1982, 1.5961,
            0.6943,  131.5327, 2.0262,   2.0509,   3.5130,
            4.0541,  1.3423,   5.2734,   32.1134,  10.5383,
            38.9974, 10.5585,  286.5404, 198.6777])
        x_std = np.array([
            17.6494, 3.0163,  0.6895,   24.2988, 16.6459,
            14.0771, 4.7035,  11.0158,  3.7845,  3.1567,
            6.2684,  0.0710,  9.1087,   3.3971,  430.3638,
            19.0690, 81.7152, 2.3992,   4.9761,  2.0648,
            1.9926,  45.4816, 1.6008,   0.3793,  1.3092,
            0.5844,  2.5511,  20.4142,  6.4362,  2.2302,
            29.8928, 7.0606,  137.3886, 96.8997])
        c_mean = np.array([60.8711, 0.5435, 0.0615, 0.0727, -59.6769, 28.4551])
        c_std = np.array([16.1887, 0.4981, 0.7968, 0.8029, 160.8846, 29.5367])

        x = data[-1, 0:34]
        c = data[-1, 34:40]
        x_norm = np.nan_to_num((x - x_mean) / x_std)
        c_norm = np.nan_to_num((c - c_mean) / c_std)

        beta = np.array([
            0.1806,  0.0249, 0.2120,  -0.0495, 0.0084,
            -0.0980, 0.0774, -0.0350, -0.0948, 0.1169,
            0.7476,  0.0323, 0.0305,  -0.0251, 0.0330,
            0.1424,  0.0324, -0.1450, -0.0594, 0.0085,
            -0.0501, 0.0265, 0.0794,  -0.0107, 0.0225,
            0.0040,  0.0799, -0.0287, 0.0531,  -0.0728,
            0.0243,  0.1017, 0.0662,  -0.0074, 0.0281,
            0.0078,  0.0593, -0.2046, -0.0167, 0.1239])
        rho = 7.8521
        nu = 1.0389

        xstar = np.concatenate((x_norm, c_norm))
        exp_bx = np.exp(np.dot(xstar, beta))
        l_exp_bx = pow(4 / rho, nu) * exp_bx

        score = 1 - np.exp(-l_exp_bx)
        label = score > 0.45

        return score, label

    def predict_patient_baseline(self, x):
        length = x.shape[0]
        scores = np.zeros(length)
        labels = np.zeros(length)
        for t in range(length):
            current_data = x[:t+1]
            current_score, current_label = \
                 self.get_sepsis_score_baseline(current_data)
            scores[t] = current_score
            labels[t] = current_label
        return scores, labels

    def predict_patient(self, x):
        x = self.preprocess_data_test(x)
        # TODO: expand data here
        length = x.shape[0]
        scores = np.zeros(length)
        labels = np.zeros(length)
        for t in range(length):
            current_data = x[:t+1]
            current_score, current_label = self.get_sepsis_score(current_data)
            scores[t] = current_score
            labels[t] = current_label
        return scores, labels

        


        
