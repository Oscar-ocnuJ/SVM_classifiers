import numpy as np
from matplotlib import pyplot as plt
from skimage import io, exposure
from skimage.transform import resize
from skimage.morphology import dilation, square
from digit import Digit


class RealDataset:
    def __init__(self, n_images=40):
        self.width = 14
        self.n_images = n_images
        self.data = dict()
        self.import_images()
        self.X = self.data['data']
        self.y = self.data['targets']
        self.digits = []
        self.create_digits()

    def create_digits(self):
        for i in range(self.n_images):
            self.digits.append(Digit(self.X[i], self.y[i]))

    def import_images(self):
        self.data['data'] = list()
        self.data['targets'] = list()
        for i in range(self.n_images):
            path = 'RealSet/' + str(i) + '.png'

            # Loading the image
            image = io.imread(path, as_gray=True)

            # Changing the size to meet the classifier number of parameters
            image = -resize(image, (self.width, self.width), anti_aliasing=False)

            # Mapping the image from 0 to 1 with a lineal function
            image = exposure.rescale_intensity(image)

            # Thresholding the image to accentuate the white background
            threshold = 0.10
            image = np.where(image > threshold, image, 0)

            # Transform the matrix(image) into a 1d vector
            image = image.reshape((self.width*self.width,))

            # Storing the vectors and targets
            self.data['data'].append(image)
            self.data['targets'].append(str(i % 10))

    def show_image(self, index):
        image = self.X[index]
        image = image.reshape(self.width, self.width)
        plt.figure()
        plt.gray()
        plt.matshow(image)
        plt.show()

    def print_info(self):
        from collections import Counter

        c = Counter(self.y)
        info = "Test dataset size " + str(self.n_images)
        key_value = {}
        for i in sorted(c.keys()):
            key_value[i] = c[i]

        plt.bar(key_value.keys(), key_value.values())
        plt.xlabel('Labels')
        plt.ylabel('Occurrence')
        plt.title('Occurrence of real dataset labels')
        ax = plt.axes()
        ax.grid(which='major', axis='y')
        plt.show()
        return info
