# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 15:29:46 2020

@author: swati
"""
import os
import argparse
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# from sklearn.preprocessing import normalize
# from feature_extrator import *


def parse_args():
    parser = argparse.ArgumentParser(description="SVM Training")
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--hinge_features", type=str)
    parser.add_argument("--cold_features", type=str)
    parser.add_argument("--gt_label", type=str)
    opt = parser.parse_args()
    return opt


def save_classifier(clf, filename):
    filename = os.path.join(".", "classifiers", filename)
    with open(filename, "wb") as clf_file:
        pickle.dump(clf, clf_file)


def svm_train(features, labels):
    x = np.load(features)
    y = np.load(labels)["label"]
    # x = normalize(x, axis=0)
    X_data, X_test, y_data, y_test = train_test_split(x, y, test_size=0.2)
    clf = SVC(kernel="rbf", C=10, probability=True)
    clf.fit(X_data, y_data)
    y_pred = clf.score(X_test, y_test)
    name = f"svm_{features.split('/')[-1].split('.')[0]}_train.pkl"
    print(features, name)
    save_classifier(clf, name)
    return y_pred


def avg_hing():
    x = 0
    for i in range(1, 101):
        y = svm_train("./features/hinge_features.npy_features.npy", "./features/labels.npz")
        print(f"{i} --> {y}")
        x += y
    print(x / 100)


def avg_cold():
    x = 0
    for i in range(1, 101):
        y = svm_train("./features/cold_features.npy", "./features/labels.npz")
        print(f"{i} --> {y}")
        x += y
    print(x / 100)


if __name__ == "__main__":
    # opt = parse_args()
    # avg_cold()
    print("[*] Training SVM -- Average Hing")
    avg_hing()
