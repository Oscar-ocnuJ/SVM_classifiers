import math
import matplotlib.pyplot as plt
from digit import Digit
from sklearn.preprocessing import MinMaxScaler
from skimage.transform import warp, AffineTransform
import cv2
import numpy as np
from torchvision import transforms


def save_image(filename):
    filepath = 'figures/' + filename + '.eps'
    if not os.path.exists(filepath):
        plt.savefig(filepath, format='eps')


class Dataset:
    def __init__(self, data, size=0, transform=False):

        self.length = int((len(data['data'])))
        if 0 < size < self.length:
            self.length = size
        else:
            size = self.length

        self.width = 28

        targets = data['target'][0:size]
        self.targets = targets
        self.data = data['data'].to_numpy()[0:size]

        self.transform = transform

        self.scaler = MinMaxScaler()
        self.scaler.fit(self.data)

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
        filename = 'occurence_mnist_labels'
        save_image(filename)
        plt.show()
        return info

    def separate_train_test(self, test_size_ratio):
        from sklearn.model_selection import train_test_split

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.targets,
                                                                                test_size=test_size_ratio)
        # Data normalization
        # self.X_train = self.X_train / 255
        # self.X_test = self.X_test / 255
        self.X_train = self.scaler.transform(self.X_train)
        if self.transform:
            random_choice = np.random.randint(0, 3)
            for i in range(2000):
                image = self.X_train[i].reshape((self.width, self.width))
                if random_choice == 0:
                    rotate_object = RandomRotate()
                    image = rotate_object(image)
                if random_choice == 1:
                    horizontal_translate_object = RandomHorizontalTranslate()
                    image = horizontal_translate_object(image)
                if random_choice == 2:
                    vertical_translate_object = RandomVerticalTranslate()
                    image = vertical_translate_object(image)
                if random_choice == 3:
                    zoom_object = RandomZoom()
                    image = zoom_object(image)
                self.X_train[i] = image.reshape((self.width*self.width,))

        self.X_test = self.scaler.transform(self.X_test)

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
        filename = 'occurrence_training_testing_sets'
        save_image(filename)
        plt.show()


class RandomRotate(object):
    """Rotate randomly"""

    def __init__(self):
        self.angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

    def __call__(self, image):
        image = image.copy()
        arg = np.random.randint(0, 4)
        image = warp(image, AffineTransform(rotation=self.angles[arg]), mode='reflect')
        return image


class RandomHorizontalTranslate(object):
    """Translate Horizontally randomly"""

    def __call__(self, image):
        image = image.copy()
        steps = [image.shape[0] // 8, image.shape[0] // 4, image.shape[0] // 2, 0]
        arg = np.random.randint(0, 4)
        image = warp(image, AffineTransform(translation=(steps[arg], 0)), mode='reflect')
        return image


class RandomVerticalTranslate(object):
    """ Translate Vertically randomly"""

    def __call__(self, image):
        image = image.copy()
        steps = [image.shape[1] // 8, image.shape[1] // 4, image.shape[1] // 2, 0]
        arg = np.random.randint(0, 4)
        image = warp(image, AffineTransform(translation=(0, steps[arg])), mode='reflect')
        return image


class RandomZoom(object):
    """Zoom randomly"""

    def __call__(self, image):
        image = image.copy()

        random_zoom_factor = 0.1 * np.random.randint(7, 12)

        height, width = image.shape[:2]  # It is also the final desired size
        new_height, new_width = int(height * random_zoom_factor), int(width * random_zoom_factor)

        # # Crop only the part that will remain in the result (more efficient)
        # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
        y1, x1 = max(0, new_height - height), max(0, new_width - width) // 2
        y2, x2 = y1 + height, x1 + width
        bbox = np.array([y1, x1, y2, x2])

        # Map back to original image coordinates
        bbox = (bbox / random_zoom_factor).astype(int)
        y1, x1, y2, x2 = bbox
        cropped_img = image[y1:y2, x1:x2]

        # Handle padding when downscaling
        resize_height, resize_width = min(new_height, height), min(new_width, width)
        pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) // 2
        pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
        pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0, 0)] * (image.ndim - 2)

        # Resize the cropped image
        image = cv2.resize(cropped_img, (resize_width, resize_height))
        image = np.pad(image, pad_spec, mode='constant')
        assert image.shape[0] == height and image.shape[1] == width

        return image


train_transform = transforms.Compose([
    RandomZoom(),
    RandomRotate(),
    RandomVerticalTranslate(),
    RandomHorizontalTranslate()
])
