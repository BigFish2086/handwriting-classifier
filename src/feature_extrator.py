# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 22:23:46 2020

@author: swati
"""
from hinge_feature_extraction import *
from cold_feature_extraction import *
import argparse
import numpy as np
import os
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-in", "--input_folder", type=str, default=r"<IMAGES FOLDER IMAGES ARE STORED>")
    parser.add_argument("-out", "--output_folder", type=str, default=r"<FOLDER WHERE FEATURES ARE STORED")
    parser.add_argument("--sharpness_factor", type=int, default=10)
    parser.add_argument("--bordersize", type=int, default=3)
    parser.add_argument("--show_images", type=bool, default=False)
    parser.add_argument("--is_binary", type=bool, default=True)
    opt = parser.parse_args()
    return opt


def image_extract_hing_feat(img_path, show_images=False, is_binary=True, bordersize=3, sharpness_factor=10):
    hinge = Hinge(show_images, is_binary, bordersize, sharpness_factor)
    h_f = hinge.get_hinge_features(img_path)
    return h_f


def image_extract_cold_feat(img_path, show_images=False, is_binary=True, bordersize=3, sharpness_factor=10):
    cold = Cold(show_images, is_binary, bordersize, sharpness_factor)
    c_f = cold.get_cold_features(img_path)
    return c_f


def extract_feat(input_folder, sharpness_factor=10, bordersize=3, show_images=False, is_binary=True):
    class_dirs = os.listdir(input_folder)

    class_dirs.sort()
    print(class_dirs)

    hinge_feature_vectors = []
    cold_feature_vectors = []
    labels = []
    label_names = []
    ecount = 0

    cold = Cold(show_images, is_binary, bordersize, sharpness_factor)
    hinge = Hinge(show_images, is_binary, bordersize, sharpness_factor)

    for i, class_dir in enumerate(class_dirs):
        img_filenames = os.listdir(os.path.join(input_folder, class_dir))
        for img_filename in tqdm(img_filenames):
            try:
                img_path = os.path.join(input_folder, class_dir, img_filename)

                h_f = hinge.get_hinge_features(img_path)
                c_f = cold.get_cold_features(img_path)

                hinge_feature_vectors.append(h_f)
                cold_feature_vectors.append(c_f)
                label_names.append(class_dir)
                labels.append(i)
            except Exception as inst:
                ecount += 1
                if ecount % 20 == 0:
                    print(inst, f"error count: {ecount}")
                continue

        print(f"[STATUS] processed folder: {class_dir}")

    return hinge_feature_vectors, cold_feature_vectors, labels, label_names


def save_feat(output_folder, hinge_feature_vectors, cold_feature_vectors, labels, label_names):
    np.save(os.path.join(output_folder, "hinge_features.npy"), hinge_feature_vectors)
    np.save(os.path.join(output_folder, "cold_features.npy"), cold_feature_vectors)
    np.savez(os.path.join(output_folder, "labels"), label=labels, label_name=label_names)
    print("Saved all hinge and cold features")


if __name__ == "__main__":

    # Parse arguments
    # opt = parse_args()

    # hinge_feature_vectors, cold_feature_vectors, labels, label_names = extract_feat(
    #     opt.input_folder, opt.sharpness_factor, opt.bordersize, opt.show_images, opt.is_binary
    # )
    # save_feat(opt.output_folder, hinge_feature_vectors, cold_feature_vectors, labels, label_names)

    hinge_feature_vectors, cold_feature_vectors, labels, label_names = extract_feat("./preprocessed/data-set/")
    save_feat("./out", hinge_feature_vectors, cold_feature_vectors, labels, label_names)
