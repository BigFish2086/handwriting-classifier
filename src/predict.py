#!env python
import os
import glob
import time
import pickle
import argparse

from utils import preprocess, pre
from feature_extrator import image_extract_hing_feat, image_extract_cold_feat


def main():
    parser = argparse.ArgumentParser(description="A male/female handwriting classifier.")
    parser.add_argument(
        "-i", "--inputdir", help="The path to the input directory. Which images are read from.", default="test"
    )
    parser.add_argument(
        "-o",
        "--outputdir",
        help="The path to the output directory. Where results and times will be reported.",
        default="out",
    )
    parser.add_argument("-g", "--hinge", action="store_true", help="Train Hinge")
    parser.add_argument("-c", "--cold", action="store_true", help="Train Cold")

    args = parser.parse_args()

    # Load the classifier
    clf_filename = ""
    extractor = None

    if args.hinge and args.cold:
        print("Use only one classifier. Please choose one of them.")
        exit(1)
    elif not args.hinge and not args.cold:
        print("Use one classifier. Use -g or -c")
        exit(1)
    elif args.hinge:
        clf_filename = os.path.join(".", "classifiers", "svm_hinge_features_train.pkl")
        extractor = image_extract_hing_feat
    elif args.cold:
        clf_filename = os.path.join(".", "classifiers", "svm_cold_features_train.pkl")
        extractor = image_extract_cold_feat

    times = []
    results = []
    test_images = sorted(glob.glob(os.path.join(args.inputdir, "*.jpg")))
    print("Testing on {} images.".format(len(test_images)))

    with open(clf_filename, "rb") as clf_file:
        clf = pickle.load(clf_file)

    print("Classifier loaded.")

    for test_image in test_images:
        start_time = time.time()
        try:
            x_cut_percent = 0.1
            y_cut_percent = 0.5
            gkernel = (9, 9)
            thresh_block_size = 101
            thresh_c = 30
            dilation_size = (15, 20)
            dilation_iterations = 8

            # Preprocess the images
            preprocess(
                test_image,
                x_cut_percent,
                y_cut_percent,
                gkernel,
                thresh_block_size,
                thresh_c,
                dilation_size,
                dilation_iterations,
                show_steps=False,
            )
            print("Preprocessing done.")

            # Extract features and Predict
            test_image_path = pre(test_image)
            print("Extracting features from {}".format(test_image_path))

            features = extractor(test_image_path)
            features = features.reshape(1, -1)
            print(features.shape, features.shape)
            print("Features extracted.")

            y_pre = clf.predict(features)[0]
            print(f"Prediction done. It's {y_pre}")
            results.append(y_pre)

        except Exception as e:
            print("ERROR", e)
            results.append(-1)

        end = time.time()
        duration = end - start_time
        if duration < 0.01:
            duration = 0.01
        times.append(f"{duration:.2f}")

    with open(os.path.join(args.outputdir, "results.txt"), "w") as results_file:
        results = [str(x) for x in results]
        results_file.write("\n".join(results))

    with open(os.path.join(args.outputdir, "times.txt"), "w") as times_file:
        times = [str(x) for x in times]
        times_file.write("\n".join(times))


if __name__ == "__main__":
    main()
