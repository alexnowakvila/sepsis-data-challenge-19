#!/usr/bin/env python

import numpy as np, os, sys
import matplotlib.pyplot as plt
import pandas as pd
import pdb
# from get_sepsis_score import load_sepsis_model, get_sepsis_score
# from utils import fill_NaN, preprocessing
from data_utils import Data
from terminaltables import AsciiTable

from sklearn.model_selection import train_test_split, KFold
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
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score, \
    average_precision_score, balanced_accuracy_score, f1_score, \
    precision_score, recall_score

from evaluate_sepsis_score import evaluate_sepsis_score_bis \
    as evaluate_sepsis_score
from models import *
import xgboost as xgb

def print_table(res_method, res_baseline):
    headers = [" ", "AUROC", "AUPRC", "ACCURACY", "F1", "UTILITY:"]
    res_method = ["method"] + ["{:.6f}".format(res) for res in res_method]
    res_baseline = ["baseline"] + \
        ["{:.6f}".format(res) for res in res_baseline]
    table_data = [headers, res_method, res_baseline]
    table = AsciiTable(table_data)
    print(table.table)

def test(model, data_x, data_y):
    num_files = len(data_x)
    Scores, Labels = [], []
    Scores_b, Labels_b = [], []
    for i, (x, y_true) in enumerate(zip(data_x, data_y)):

        # model
        scores, labels = model.predict_patient(x)
        Scores.append(scores)
        Labels.append(labels)

        # baseline
        scores, labels = model.predict_patient_baseline(x)
        Scores_b.append(scores)
        Labels_b.append(labels)

        if i % 200 == 0 and i > 0:
            print('    {}/{}...'.format(i, num_files))
            # evaluate method and baseline
            res_method = evaluate_sepsis_score(data_y[:i+1], Scores, Labels)
            res_baseline = evaluate_sepsis_score(data_y[:i+1], \
                Scores_b, Labels_b)
            print_table(res_method, res_baseline)


# def subsample_dataset(X_train, y_train, num=5000):
#     # find positive examples
#     y_train_lab = np.array([y.max() for y in y_train])
#     indices_0 =  y_train[y_train_lab == 0]
#     indices_1 =  y_train[y_train_lab == 1]

#     pdb.set_trace()

def remove_nosepsis(X_train, y_train):
    n_train = len(X_train)
    X_train_n = []
    y_train_n = []
    for i, y in enumerate(y_train):
        if y.max() == 1:
            X_train_n.append(X_train[i])
            y_train_n.append(y_train[i])
    return X_train_n, y_train_n



if __name__ == '__main__':
    np.random.seed(42)

    ###########################################################################
    # load data
    ###########################################################################

    # Load dataset(s)
    # directories = ["training_setA"]
    # directories = ["training_setB"]
    directories = ["training_setB", "training_setA"]
    # directories = ["training_setA"]
    data = Data()
    data.load_data(directories)
    # data.load_data(directories, top=1000)

    ###########################################################################
    # data pre-analysis
    ###########################################################################

    # Compute basic statistics
    data.basic_statistics()
    print("\n Statistics of DataSet A \n", data.stats)
    # visualizations
    # data.plot_fill_counts(show=True)
    # data.plot_proportion_sepsis(show=True)
    # data.plot_proportion_sepsis_sepsispatients(show=True)
    # data.plot_hospitalization_time(show=True)

    ###########################################################################
    # data pre-processing
    ###########################################################################

    data.add_feature(0)
    data.fill_NaN(0)
    data.fill_NaN(1)
    
    
    ###########################################################################
    # split data
    ###########################################################################

    X, y = data.data, data.labels
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=42)

    # remove no sepsis patients from training data
    # X_train, y_train = remove_nosepsis(X_train, y_train)

    classifier_names = [
        # "Random Forest",
        # "Neural Net", 
        # "AdaBoost",
        # "Logistic Regression",
        # "SVC",
        # "XgBoost",
        "Naive Bayes",
        #"QDA"]
        ]
    classifiers = [
        # RandomForestClassifier(n_estimators=10),
        # MLPClassifier(alpha=1, max_iter=1000),
        # AdaBoostClassifier(),
        # LogisticRegression(penalty='l2', C=1e5, max_iter=1000),
        # SVC(kernel="linear", C=0.025),
        # xgb.XGBClassifier(random_state=42),
        GaussianNB(),
        #QuadraticDiscriminantAnalysis()]
        ]

    for i, clf in enumerate(classifiers):
        print('------ {} ------'.format(classifier_names[i]))
        model = IndependentModel(classifiers[i], classifier_names[i], k=0)
        # train model
        model.fit(X_train, y_train)
        # test model
        test(model, X_test, y_test)
        # save model
        # TODO