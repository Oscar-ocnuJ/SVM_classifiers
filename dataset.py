import math
import numpy as np
import matplotlib.pyplot as plt
from digit import Digit
from sklearn.preprocessing import StandardScaler
import cv2
import pickle


class Dataset:
    def __init__(self, data, size=0):

        self.length = int((len(data['data'])))
        if 0 < size < self.length:
            self.length = size
        else:
            size = self.length
        self.scaler = StandardScaler()
        self.targets = data['target'][0:size]
        self.data = data['data'].to_numpy()[0:size]
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

    def zoom_in_images(self, data):
        # Zoom from 0.5x to 1.2x
        zoom_factors = [round(0.1*x, 2) for x in range(5, 13)]
        factor = 1

        img = data[1]

        width = img.shape[0]  # It is also the final desired size
        new_width = int(width * factor)

        # # Crop only the part that will remain in the result (more efficient)
        # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
        y1 = max(0, new_width - width) // 2
        x1, y2, x2 = y1 + width
        bbox = np.array([y1, x1, y2, x2])

        # Map back to original image coordinates
        bbox = (bbox / factor).astype(int)
        y1, x1, y2, x2 = bbox
        cropped_img = img[y1:y2, x1:x2]

        # Handle padding when downscaling
        resize_width = min(new_width, width)
        pad_height1, pad_width1 = (width - resize_width) // 2
        pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
        pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0, 0)] * (img.ndim - 2)

        result = cv2.resize(cropped_img, (resize_width, resize_height))
        result = np.pad(result, pad_spec, mode='constant')
        assert result.shape[0] == height and result.shape[1] == width
        toto taga
        return result

    def scaler_fit_transform(self, data):
        self.scaler.fit_transform(data)

    def scaler_fit(self, data):
        self.scaler.fit(data)
    def dump_scaler(self, filename='scaler.pkl'):
        pickle.dump(self.scaler, filename)

    def load_scaler(self, filename='scaler.pkl'):
        self.scaler = pickle.load(filename)
