import math
import time

import numpy as np
import matplotlib.pyplot as plt
from digit import Digit
from sklearn.preprocessing import StandardScaler
import cv2
import pickle


def zoom_in_out(image, zoom_factor):

    # If the zoom_factor equals to 1, return the same image
    if zoom_factor == 1:
        return image

    height, width = image.shape[:2]  # It is also the final desired size
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    # # Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height), max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1, x1, y2, x2])

    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(int)
    y1, x1, y2, x2 = bbox
    cropped_img = image[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) // 2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0, 0)] * (image.ndim - 2)

    # Resize the cropped image
    zoomed_image = cv2.resize(cropped_img, (resize_width, resize_height))
    zoomed_image = np.pad(zoomed_image, pad_spec, mode='constant')
    assert zoomed_image.shape[0] == height and zoomed_image.shape[1] == width

    return zoomed_image


class Dataset:
    def __init__(self, data, size=0, apply_zoom=False):

        self.length = int((len(data['data'])))
        if 0 < size < self.length:
            self.length = size
        else:
            size = self.length
        self.width = int(math.sqrt(data.data.shape[1]))
        self.scaler = StandardScaler()
        self.targets = data['target'][0:size]
        self.data = data['data'].to_numpy()[0:size]

        if apply_zoom:
            t0 = time.time()
            n_images_to_zoom = 1000
            zoom_factors = [round(0.1*x, 2) for x in range(4, 4 + int(n_images_to_zoom/100))]
            zoom_factor = zoom_factors[0]
            for image, i in zip(self.data, range(n_images_to_zoom)):
                if (i % 100) == 0:
                    zoom_factor = zoom_factors[int(i/100)]
                    print(zoom_factor, i)
                zoomed_image = zoom_in_out(image.reshape((self.width, self.width)), zoom_factor)
                self.data[i] = zoomed_image.reshape((self.width*self.width))

            print(f"Zooming {n_images_to_zoom} images has taken {1000*(time.time() - t0)} ms.")

        self.digits = []
        self.create_digits()
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

    def create_digits(self):
        for i in range(self.length):
            self.digits.append(Digit(self.data[i], self.targets[i]))

    def print_info(self):
        from collections import Counter

        c = Counter(self.targets)
        info = "Dataset size " + str(self.length)
        key_value = {}
        for i in sorted(c.keys()):
            key_value[i] = c[i]

        ax = plt.axes()
        ax.set_axisbelow(True)
        ax.yaxis.grid()
        plt.interactive(False)
        plt.bar(key_value.keys(), key_value.values())
        plt.xlabel('Labels')
        plt.ylabel('Occurrence')
        plt.title('Occurrence of MNIST dataset labels')
        plt.show()
        return info

    def separate_train_test(self, test_size_ratio):
        from sklearn.model_selection import train_test_split

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.targets,
                                                                                test_size=test_size_ratio)
        self.X_train = self.X_train / 255
        self.X_test = self.X_test / 255
        print('Size of training set: ' + str(len(self.y_train)) + '/' + str(len(self.data)))
        print('Size of testing set: ' + str(len(self.y_test)) + '/' + str(len(self.data)))

    def display_train_test(self):
        from collections import Counter

        test = Counter(self.y_test)
        train = Counter(self.y_train)
        info = "Dataset size: " + str(self.length)
        print(info)

        key_value_train = {}
        key_value_test = {}

        for i in sorted(test.keys()):
            key_value_test[i] = test[i]
        for i in sorted(train.keys()):
            key_value_train[i] = train[i]

        ax = plt.axes()
        ax.set_axisbelow(True)
        ax.yaxis.grid()

        p1 = plt.bar(key_value_train.keys(), key_value_train.values(), width=0.75)
        p2 = plt.bar(key_value_test.keys(), key_value_test.values(), width=0.75, bottom=list(key_value_train.values()))

        plt.legend((p1[0], p2[0]), ('Training set', 'Test set'), loc='lower left')
        plt.xlabel('Labels')
        plt.ylabel('Occurrence')
        plt.title('Occurrence of training and testing sets')
        plt.show()


    def scaler_fit_transform(self, data):
        self.scaler.fit_transform(data)

    def scaler_fit(self, data):
        self.scaler.fit(data)
    def dump_scaler(self, filename='scaler.pkl'):
        pickle.dump(self.scaler, filename)

    def load_scaler(self, filename='scaler.pkl'):
        self.scaler = pickle.load(filename)
