"""
This file is used to create all the necessary directory structure
or clear the those directory
"""

import os
import shutil
import argparse


def create_dir(path):
    """
    Create the directory if it does not exist
    :param path: path to the directory
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)


def create_dir_structure():
    """
    Create the directory structure
    :return: None
    """
    # for the prediction/testing
    create_dir(os.path.join(os.getcwd(), "out"))
    create_dir(os.path.join(os.getcwd(), "test"))

    # for storing and classifying the data-set
    create_dir(os.path.join(os.getcwd(), "data-set"))
    create_dir(os.path.join(os.getcwd(), "data-set", "males"))
    create_dir(os.path.join(os.getcwd(), "data-set", "females"))

    # for extracting the features and training the classifiers
    create_dir(os.path.join(os.getcwd(), "features"))
    create_dir(os.path.join(os.getcwd(), "classifiers"))

    # for storing the preprocessed images of the data-set and the test data
    create_dir(os.path.join(os.getcwd(), "preprocessed"))
    create_dir(os.path.join(os.getcwd(), "preprocessed", "test"))
    create_dir(os.path.join(os.getcwd(), "preprocessed", "data-set"))
    create_dir(os.path.join(os.getcwd(), "preprocessed", "data-set", "males"))
    create_dir(os.path.join(os.getcwd(), "preprocessed", "data-set", "females"))


# delete a directory
def delete_dir():
    # ask the user if he/she wants to delete the directory
    print("The following directories?\n- out\n- features\n- classifiers\n- preprocessed\nDelete them? (y/n)")
    choice = input()
    if choice == "y":
        shutil.rmtree(os.path.join(os.getcwd(), "out"))
        shutil.rmtree(os.path.join(os.getcwd(), "features"))
        shutil.rmtree(os.path.join(os.getcwd(), "classifiers"))
        shutil.rmtree(os.path.join(os.getcwd(), "preprocessed"))
    else:
        print("Directory structure not deleted")


def parse_args():
    parser = argparse.ArgumentParser(description="Clean / Build the required the directory structure for the project")
    parser.add_argument("-b", "--build", action="store_true", help="Build the directory structure")
    parser.add_argument("-c", "--clean", action="store_true", help="clean the directory structure")
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_args()
    # check if two arguments are passed
    if opt.build and opt.clean:
        print("Please pass only one argument")
        exit(1)

    print(opt)

    if opt.build:
        create_dir_structure()
    elif opt.clean:
        delete_dir()
