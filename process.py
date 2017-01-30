import math, time
import pymeanshift as pms
import numpy as np
import cv2


DIR_HORIZONTAL = 0
DIR_VERTICAL = 1


def find_edges(img, img_labels):
    res = np.copy(img)

    for row in range(1, img_labels.shape[0]):
        for col in range(1, img_labels.shape[1]):
            if (img_labels[row, col] != img_labels[row - 1, col]
             or img_labels[row, col] != img_labels[row, col - 1]):
                res[row, col] = (0, 0, 255)

    return res


def find_cluster_boundaries(img, img_labels):
    main_label = img_labels[220, 270]
    boundaries = list()

    rows, cols = np.where(img_labels == main_label)

    for row in set(rows):
        col_list = cols[np.where(rows == row)]
        left, right = np.min(col_list), np.max(col_list)

        img[row, left] = (0, 0, 255)
        img[row, right] = (0, 0, 255)

        boundaries.append((row, left))
        boundaries.append((row, right))

    boundaries[:2] = []
    boundaries[-2:] = []

    top_bound, bottom_bound = np.min(rows), np.max(rows)

    top_index = np.where(rows == top_bound)
    bottom_index = np.where(rows == bottom_bound)

    for index in top_index[0]:
        boundaries.insert(0, (rows[index], cols[index]))
    for index in bottom_index[0]:
        boundaries.append((rows[index], cols[index]))

    img[rows[top_index], cols[top_index]] = (0, 0, 255)
    img[rows[bottom_index], cols[bottom_index]] = (0, 0, 255)

    return boundaries


def detect_foreground(original_img, img_labels):
    start = time.time()

    height, width = img_labels.shape
    img = np.copy(original_img)

    boundaries = find_cluster_boundaries(img, img_labels)

    print(time.time() - start)
    for bound in boundaries:
        print(bound)

    return img











