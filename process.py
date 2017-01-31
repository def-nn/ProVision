import math
import time
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
                res[row - 1: row + 2, col - 1: col + 2] = (255, 255, 255)

    return res


class ForegroundObject:
    """
    Class for keeping an information about detected object

    Args:
        center (`tuple` of `int`) : coordinates of object center (row, col)
        label               (int) : label which is definitely belongs to an object

    Attributes:
        c_row, c_col                      (int) : coordinates of object center
        main_label                        (int) : start label
        labels                (`list` of `int`) : list of labels belonging to the object
        boundaries (`tuple` of `numpy.ndarray`) : first element of a tuple contains coordinates of boundaries
                                                  second its a labels of clusters which borders the object
                                                  at the corresponding points from first array
    """

    def __init__(self, center, label):
        self.c_row, self.c_col = center
        self.main_label = label
        self.labels = [label]
        self.boundaries = []


class Detector:
    """
    Class for object detection in segmented images

    Note:
        Before usage f_object attribute must be set or KeyError will be raised.
        Could be applied multiple times with different ForegroundObject instances

    Args:
        segmented_img (numpy.ndarray) : segmented image as matrix
        img_labels    (numpy.ndarray) : labels of segmented image

    Attributes:
        img_labels    (numpy.ndarray) : labels of the processed image
        f_object   (ForegroundObject) : detected object
        hue           (numpy.ndarray) : hue channel of an image in HSV color space
        sat           (numpy.ndarray) : sat channel of an image in HSV color space
        val           (numpy.ndarray) : val channel of an image in HSV color space
    """

    def __init__(self, segmented_img, img_labels):
        self.img_labels = img_labels
        self.f_object = Detector.undefined

        img_hsv = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2HSV)

        # Normalize color properties
        #   hue:         [0; 2pi]
        #   saturation:  [0: 100]
        #   vibration:   [0; 100]
        self.hue = np.float32(img_hsv[:, :, 0] * 2 * math.pi / 180)         # Convert degrees to radians
        self.sat = np.float32(img_hsv[:, :, 1] / 255)
        self.val = np.float32(img_hsv[:, :, 2] / 255)

    def __setattr__(self, key, value):
        if key == 'f_object':
            if isinstance(value, ForegroundObject) or isinstance(value, property):
                self.__dict__[key] = value
            else:
                raise ValueError("Member f_object should be an instance of ForegroundObject class as parameter")
        else:
            self.__dict__[key] = value

    def find_color_distance(self, p1, p2):
        """
        Find euclidean distance between two pixels in HSV color space

        Note:
            To avoid a problem of passing negative value as a parameter to a sqrt() function
            which may appear if both pixels have the same values on each HSV channel (due to loss of accuracy)
            we have to check that pixels values differs or just return 0 otherwise

        :param p1: HSV values of first pixel
        :param p2: HSV values of second pixel
        """
        h1, s1, v1 = p1
        h2, s2, v2 = p2

        if h1 == h2 and s1 == s2 and v1 == v2: return 0

        return math.sqrt(
            s1 ** 2 + s2 ** 2 -
            s1 * s2 * math.cos(min(abs(h1 - h2), math.pi * 2 - abs(h1 - h2))) * 2 +
            (v1 - v2) ** 2
        )

    def find_cluster_boundaries(self, label):
        """
        Find boundary points in given cluster

        :param label: label of cluster, which boundaries have to be found
        :return: (`tuple` of `numpy.ndarray`) - first element of a tuple contains coordinates of boundaries
                                                second its a labels of clusters which borders the object
                                                at the corresponding points from first array
        """
        bound_coord = list()
        bound_label = list()

        rows, cols = np.where(self.img_labels == label)

        for row in set(rows):
            col_list = cols[np.where(rows == row)]
            left, right = np.min(col_list), np.max(col_list)

            if left != 0:
                bound_coord.append((row, left))
                bound_label.append(self.img_labels[row, left - 1])
            if right != self.img_labels.shape[1] - 1:
                bound_coord.append((row, right))
                bound_label.append(self.img_labels[row, right + 1])

        bound_coord[:2] = []
        bound_coord[-2:] = []
        bound_label[:2] = []
        bound_label[-2:] = []

        top_bound, bottom_bound = np.min(rows), np.max(rows)

        top_index = np.where(rows == top_bound)
        bottom_index = np.where(rows == bottom_bound)

        for index in top_index[0]:
            if rows[index] != 0:
                bound_coord.insert(0, (rows[index], cols[index]))
                bound_label.insert(0, self.img_labels[rows[index] - 1, cols[index]])
        for index in bottom_index[0]:
            if rows[index] != self.img_labels.shape[0] - 1:
                bound_coord.append((rows[index], cols[index]))
                bound_label.append(self.img_labels[rows[index] + 1, cols[index]])

        boundaries = (
            np.asarray(bound_coord),
            np.asarray(bound_label)
        )
        return boundaries

    def extend_obj_boundaries(self, label):
        """
        Extends object boundary with area of given cluster

        :param label: label of cluster, which boundaries have to be found
        """
        extended_bound_coord, extended_bound_label = self.find_cluster_boundaries(label)
        obj_bound_coord, obj_bound_label = self.f_object.boundaries

        new_bound_cord = np.concatenate((
            extended_bound_coord[np.where(extended_bound_label != self.f_object.main_label)],
            obj_bound_coord[np.where(obj_bound_label != label)]
        ))

        new_bound_label = np.concatenate((
            extended_bound_label[np.where(extended_bound_label != self.f_object.main_label)],
            obj_bound_label[np.where(obj_bound_label != label)]
        ))

        self.f_object.boundaries = (new_bound_cord, new_bound_label)

    def find_cluster_property(self):
        """
        :return: tuple that contains sum of all elements of the objects on each channels
                 (in HSV color space) and number of all elements
        """
        numerator_hue = numerator_sat = numerator_val = denominator = 0

        for label in self.f_object.labels:
            index = np.where(self.img_labels == label)

            numerator_hue += np.sum(self.hue[index])
            numerator_sat += np.sum(self.sat[index])
            numerator_val += np.sum(self.val[index])

            denominator += len(index[0])

        return numerator_hue, numerator_sat, numerator_val, denominator

    def compute_avg(self, cluster_property, label):
        """
        Computing average color variance in object as if elements of given cluster would be added to it.

        :param cluster_property: init properties of object - sum of all elements of the objects on each channels
                                 (in HSV color space) and number of all elements
        :param label: label of cluster, which element have to be added to object

        :return: computing variance - euclidean distance between modified average values of object and values of
                                      main object cluster in HSV color space
        """
        numerator_hue, numerator_sat, numerator_val, denominator = cluster_property

        index = np.where(self.img_labels == label)

        numerator_hue += np.sum(self.hue[index])
        numerator_sat += np.sum(self.sat[index])
        numerator_val += np.sum(self.val[index])

        denominator += len(index[0])

        avg_hue = numerator_hue / denominator
        avg_sat = numerator_sat / denominator
        avg_val = numerator_val / denominator

        return self.find_color_distance(
            (avg_hue, avg_sat, avg_val),
            (
                self.hue[self.f_object.c_row, self.f_object.c_col],
                self.sat[self.f_object.c_row, self.f_object.c_col],
                self.val[self.f_object.c_row, self.f_object.c_col]
            )
        )

    def find_neighbour_labels(self):
        """
        :return: list of labels of all clusters which adjoin to an object
        """
        neighbour = set()
        for row, col in self.f_object.boundaries[0]:

            if row != self.img_labels.shape[0] - 1 and row != 0:
                if self.img_labels[row - 1, col] not in self.f_object.labels:
                    neighbour.add(self.img_labels[row - 1, col])
                if self.img_labels[row + 1, col] not in self.f_object.labels:
                    neighbour.add(self.img_labels[row + 1, col])

            if col != self.img_labels.shape[1] - 1 and col != 0:
                if self.img_labels[row, col - 1] not in self.f_object.labels:
                    neighbour.add(self.img_labels[row, col - 1])
                if self.img_labels[row, col + 1] not in self.f_object.labels:
                    neighbour.add(self.img_labels[row, col + 1])

        return neighbour

    def detect_object(self):
        """
        :return: top-left and bottom-right coordinates of boundary rectangle of detected object
        """
        start = time.time()

        mask = np.zeros(self.img_labels.shape, dtype=np.uint8)
        mask[np.where(self.img_labels == self.f_object.main_label)] = 255

        self.f_object.boundaries = self.find_cluster_boundaries(self.f_object.main_label)

        while True:
            neighbours = self.find_neighbour_labels()

            exit_flag = True

            for label in neighbours:
                cluster_property = self.find_cluster_property()
                if self.compute_avg(cluster_property, label) < 0.1:
                    self.f_object.labels.append(label)
                    self.extend_obj_boundaries(label)
                    mask[np.where(self.img_labels == label)] = 255
                    exit_flag = False

            if exit_flag: break

        image, contours, hierarcly = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])

        print(time.time() - start)

        return (x, y), (x + w, y + h)

    @staticmethod
    def get_undefined_object(self):
        raise KeyError("You have to define f_object member with instance of ForegroundObject")

    undefined = property(get_undefined_object)
