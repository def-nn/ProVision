import math
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











