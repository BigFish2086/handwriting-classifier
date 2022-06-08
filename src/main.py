#!/usr/bin/env python
# coding: utf-8

import glob
import time
import argparse
from tqdm import tqdm

# extract both the hing and cold features
from utils import *
from feature_extrator import *
from svm_train import *


def preprocess_data():
    all_images = glob.glob("./data-set/*/*")
    print("Preprocessing data...")
    print("Total images: ", len(all_images))
    start_time = time.time()

    x_cut_percent = 0.1
    y_cut_percent = 0.5
    gkernel = (9, 9)
    thresh_block_size = 101
    thresh_c = 30
    dilation_size = (15, 20)
    dilation_iterations = 8

    for image_path in tqdm(all_images):
        preprocess(
            image_path,
            x_cut_percent,
            y_cut_percent,
            gkernel,
            thresh_block_size,
            thresh_c,
            dilation_size,
            dilation_iterations,
        )

    end_time = time.time()
    print(f"Preprocessing time: {end_time - start_time:.2f}\n")


def extract_features():
    print("Extracting features...")
    start_time = time.time()
    hinge_feature_vectors, cold_feature_vectors, labels, label_names = extract_feat("./preprocessed/data-set/")
    save_feat("./features", hinge_feature_vectors, cold_feature_vectors, labels, label_names)
    end_time = time.time()
    print(f"Feature extraction time: {end_time - start_time:.2f}\n")


def train_hing():
    print("Training Hinge...")
    start_time = time.time()
    hing = svm_train(feat("hinge_features.npy"), feat("labels.npz"))
    end_time = time.time()
    print(f"Hing Accuracey: {hing}")
    print(f"Hinge training time: {end_time - start_time:.2f}\n")


def train_cold():
    print("Training Cold...")
    start_time = time.time()
    cold = svm_train(feat("cold_features.npy"), feat("labels.npz"))
    end_time = time.time()
    print(f"Cold Accuracey: {cold}")
    print(f"Cold training time: {end_time - start_time:.2f}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SVM classifier")
    parser.add_argument("-p", "--preprocess", action="store_true", help="Preprocess data")
    parser.add_argument("-e", "--extract", action="store_true", help="Extract features")
    parser.add_argument("-g", "--hinge", action="store_true", help="Train Hinge")
    parser.add_argument("-c", "--cold", action="store_true", help="Train Cold")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.preprocess and not args.extract and not args.hinge and not args.cold:
        print("No arguments provided. Use -h for help")
        exit(1)

    if args.preprocess:
        preprocess_data()
    if args.extract:
        extract_features()
    if args.hinge:
        train_hing()
    if args.cold:
        train_cold()
