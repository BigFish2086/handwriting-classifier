import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import functools


# check if the file passed exists
def check_file(filepath):
    return os.path.isfile(filepath)


def pre(image_path: str):
    # remove the first ./
    image_path = image_path.replace("./", "")
    return f"./preprocessed/{image_path}"


def feat(image_path: str):
    return f"./features/{image_path}"


# Show the figures / plots inside the notebook
def show_images(images, titles=None):
    # This function is used to show image(s) with titles by sending
    # an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None:
        titles = ["(%d)" % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis("off")
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()
