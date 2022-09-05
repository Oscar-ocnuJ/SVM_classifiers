import numpy as np
import matplotlib.pyplot as plt


def compute_entropy(image):
    len_sig = image.size
    sym_set = list(set(image))
    pro_pab = [np.size(image[image == i]) / (1.0 * len_sig) for i in sym_set]
    ent = np.sum([p * np.log2(1.0 / p) for p in pro_pab])
    return ent


class Digit:
    def __init__(self, data, target):
        self.width = 14
        self.target = target
        self.image = data
        self.features = {'var': 0.0, 'std': 0.0,
                         'mean': 0.0, 'entropy': 0.0}
        self.compute_features()

    def compute_features(self):
        self.features['var'] = round(np.var(self.image), 2)
        self.features['std'] = round(np.std(self.image), 2)
        self.features['mean'] = round(np.mean(self.image), 2)
        self.features['entropy'] = round(compute_entropy(self.image), 2)

    def print(self):
        print("Digit target: " + str(self.target))
        print("Digit target size: " + str(self.width) + "x" + str(self.width) +
              '| mean : ' + str(self.features['mean']) +
              '| var : ' + str(self.features['var']) +
              '| std :' + str(self.features['std']) +
              '| entropy :' + str(self.features['entropy']))
        print("Digit image:")
        plt.figure()
        plt.gray()
        plt.matshow(self.image.reshape(self.width, self.width))
        # plt.savefig(str(self.target) + '.png', bbox_inches='tight')
        plt.show()
