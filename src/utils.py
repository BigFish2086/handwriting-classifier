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


def crop_xy(img, x_cut_percent=None, y_cut_percent=None):
    height, width = img.shape
    if x_cut_percent:
        start_x, end_x = int(width / (100 / x_cut_percent)), width - int(width / (100 / x_cut_percent))
        img = img[:, start_x:end_x]

    if y_cut_percent:
        start_y, end_y = int(height / (100 / y_cut_percent)), height - int(height / (100 / y_cut_percent))
        img = img[start_y:end_y, :]
    return img


def blur_dilate(img, gkernel, thresh_block_size, thresh_c, dilation_size, dilation_iterations):
    img = cv2.GaussianBlur(img, gkernel, 0)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, thresh_block_size, thresh_c
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilation_size)
    img = cv2.dilate(img, kernel, iterations=dilation_iterations)
    return img


def get_biggest_contour(img):
    # img, mode, method
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    biggest_contour = functools.reduce(lambda c1, c2: c1 if cv2.contourArea(c1) > cv2.contourArea(c2) else c2, contours)
    return biggest_contour


def get_text_area(img, thresh_block_size, thresh_c, dilation_size, dilation_iterations, biggest_contour):
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, thresh_block_size, thresh_c
    )
    img = cv2.threshold(img, 255 / 2, 255, cv2.THRESH_BINARY)[1]
    x, y, w, h = cv2.boundingRect(biggest_contour)
    img = img[y : y + h, x : x + w]
    return img


def preprocess(
    image_path,
    x_cut_percent,
    y_cut_percent,
    gkernel,
    thresh_block_size,
    thresh_c,
    dilation_size,
    dilation_iterations,
    show_steps=False,
):
    to_preview = []

    # step 1. read the image, crop it from the x and y axis if needed
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = crop_xy(original, x_cut_percent, y_cut_percent)
    to_preview.append(img)

    # step 2. smoothen the image using guassian kernal,
    # make the image binary, then dilate it to have blocks of `whites`
    img2 = blur_dilate(img, gkernel, thresh_block_size, thresh_c, dilation_size, dilation_iterations)
    to_preview.append(img2)

    # step 3. get the biggest rounding contour which should be that
    # sorrounding the text the has been just dilated
    biggest_contour = get_biggest_contour(img2)

    # step 4. get that part of the binary image with tha rounding contour around it
    img3 = get_text_area(img, thresh_block_size, thresh_c, dilation_size, dilation_iterations, biggest_contour)
    to_preview.append(img3)

    if show_steps:
        show_images(to_preview)

    cv2.imwrite(pre(image_path), img3)

    return img3


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
