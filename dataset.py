import matplotlib.pyplot as plt
from digit import Digit
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize
import math
import numpy as np


class Dataset:
    def __init__(self, data, size=0):

        self.length = int((len(data['data'])))
        if 0 < size < self.length:
            self.length = size
        else:
            size = self.length
        self.scaler = StandardScaler()
        self.targets = data['target'][0:size]
        data = data['data'].to_numpy()[0:size]

        # Changing the size of the images if desired

        self.width = 14
        if self.width != 28:
            data_resized = np.ndarray((size, self.width*self.width))
            size = int(math.sqrt(data.shape[1]))
            for image, i in zip(data, range(data.shape[0])):
                image = image.reshape((size, size))
                image = resize(image, (self.width, self.width), anti_aliasing=True)
                data_resized[i] = image.reshape((self.width*self.width))
            self.data = data_resized
        else:
            self.data = data
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
        plt.interactive(False)
        plt.bar(key_value.keys(), key_value.values())
        plt.xlabel('Labels')
        plt.ylabel('Occurrence')
        plt.title('Occurrence of MNIST dataset labels')
        ax = plt.axes()
        ax.grid(which='major', axis='y')
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

        p1 = plt.bar(key_value_train.keys(), key_value_train.values(), width=0.5)
        p2 = plt.bar(key_value_test.keys(), key_value_test.values(), width=0.5, bottom=list(key_value_train.values()))

        plt.legend((p1[0], p2[0]), ('Training set', 'Test set'), loc='lower left')
        plt.xlabel('Labels')
        plt.ylabel('Occurrence')
        plt.title('Occurrence of training and testing sets')
        ax = plt.axes()
        ax.grid(which='major', axis='y')
        plt.show()
