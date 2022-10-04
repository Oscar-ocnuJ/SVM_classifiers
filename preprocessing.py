import time

from sklearn import svm
from real_dataset import RealDataset
from tools import import_mnist_dataset


class Kernel:
    def __init__(self, name, n_samples):
        self.name = name
        self.n_samples = n_samples
        self.accuracy_train_test = None
        self.accuracy_test_set = None
        self.accuracy_own_set = None
        self.processing_time = None


class TestPreprocessing:
    def __init__(self):
        kernels = list()
        kernels.append(Kernel('LinearSVC', 1000))
        kernels.append(Kernel('linear', 1000))
        kernels.append(Kernel('poly', 2000))
        kernels.append(Kernel('rbf', 12000))
        kernels.append(Kernel('sigmoid', 1000))
        self.kernels = kernels

        # Launch test
        self.preprocessing()

    def preprocessing(self):
        real_dataset = RealDataset(80)
        for kernel, i in zip(self.kernels, range(len(self.kernels))):
            print(100 * '-')
            print('Testing a ' + kernel.name + ' SCV classifier...')
            if kernel.name == 'LinearSVC':
                clf = svm.LinearSVC()
            else:
                clf = svm.SVC(kernel=kernel.name)

            # Importing dataset
            dataset = import_mnist_dataset(kernel.n_samples, transform=True)

            # Splitting the dataset into train and test set
            test_size_ratio = 0.2
            dataset.separate_train_test(test_size_ratio)

            # Fitting the model
            t = time.process_time()
            clf.fit(dataset.X_train, dataset.y_train)

            # Results in the dataset
            accuracy_train_set = clf.score(dataset.X_train, dataset.y_train)
            accuracy_test_set = clf.score(dataset.X_test, dataset.y_test)
            print(f"Accuracy in training set: {100*accuracy_train_set} %")
            print(f"Accuracy in test set: {100*accuracy_test_set} %")
            self.kernels[i].accuracy_train_set = accuracy_train_set
            self.kernels[i].accuracy_test_set = accuracy_test_set

            # Results in own set
            accuracy_own_set = clf.score(real_dataset.X, real_dataset.y)
            print(f"Accuracy in own set: {100*accuracy_own_set} %")
            self.kernels[i].accuracy_own_set = accuracy_own_set

            # Processing time
            processing_time = time.process_time() - t
            print(f"Processing time: {processing_time} [s]")
            self.kernels[i].processing_time = processing_time


test = TestPreprocessing()
