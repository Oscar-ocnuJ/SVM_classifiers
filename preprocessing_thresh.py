import time
import os
import matplotlib.pyplot as plt
from real_dataset import RealDataset
from sklearn import svm
from tools import import_mnist_dataset, save_results, import_results


class Kernel:
    def __init__(self, name, n_samples, length):
        self.name = name
        self.n_samples = n_samples
        self.accuracy_train_set = None
        self.accuracy_test_set = None
        self.accuracies_own_set = [None]*length
        self.processing_time = None


class TestPreprocessing:
    def __init__(self):
        self.threshold_values = [0.01*x for x in range(0, 101)]

        kernels = list()
        length = len(self.threshold_values)
        kernels.append(Kernel('LinearSVC', 1_000, length))
        kernels.append(Kernel('linear', 1_000, length))
        kernels.append(Kernel('poly', 2_000, length))
        kernels.append(Kernel('rbf', 12_000, length))
        kernels.append(Kernel('sigmoid', 1_000, length))
        self.kernels = kernels

        # Fill object
        base_name = 'thresholding'
        if not os.path.exists('results/' + base_name + '.results_pkl'):
            print("Test execution...")
            self.thresholding()
            self.save()
        else:
            print("Test made previously, loading results...")
            loaded_data = import_results(base_name)
            for kernel, i in zip(self.kernels, range(len(self.kernels))):
                self.kernels[i].accuracy_train_set = loaded_data[kernel.name]['acc_train_set']
                self.kernels[i].accuracy_test_set = loaded_data[kernel.name]['acc_test_set']
                self.kernels[i].accuracies_own_set = loaded_data[kernel.name]['acc_own_set']
                self.kernels[i].processing_time = loaded_data[kernel.name]['processing_time']

        # Plot results
        self.plot_results(save=True)

    def thresholding(self):

        for kernel, i in zip(self.kernels, range(len(self.kernels))):
            print(100*'-')
            print('Testing a ' + kernel.name + ' SCV classifier...')
            if kernel.name == 'LinearSVC':
                clf = svm.LinearSVC()
            else:
                clf = svm.SVC(kernel=kernel.name)

            # Importing dataset
            dataset = import_mnist_dataset(kernel.n_samples)

            # Splitting the dataset into train and test set
            test_size_ratio = 0.2
            dataset.separate_train_test(test_size_ratio)

            # Fitting the model
            t = time.process_time()
            clf.fit(dataset.X_train, dataset.y_train)

            # Results in the dataset
            accuracy_train_set = clf.score(dataset.X_train, dataset.y_train)
            accuracy_test_set = clf.score(dataset.X_test, dataset.y_test)
            print(f"Accuracy in training set: {accuracy_train_set}")
            print(f"Accuracy in test set: {accuracy_test_set}")
            self.kernels[i].accuracy_train_set = accuracy_train_set
            self.kernels[i].accuracy_test_set = accuracy_test_set

            # Results in the own set
            for x, j in zip(self.threshold_values, range(len(self.threshold_values))):
                real_dataset = RealDataset(n_images=80, threshold=x)
                accuracy = clf.score(real_dataset.X, real_dataset.y)
                self.kernels[i].accuracies_own_set[j] = accuracy

            processing_time = round(time.process_time() - t, 4)
            print(f"Processing time: {processing_time} s")
            self.kernels[i].processing_time = processing_time

    def save(self):
        # Create a dictionary to store the data
        data_to_save = {}
        for kernel in self.kernels:
            data_to_save[kernel.name] = {}
            data_to_save[kernel.name]['acc_train_set'] = kernel.accuracy_train_set
            data_to_save[kernel.name]['acc_test_set'] = kernel.accuracy_test_set
            data_to_save[kernel.name]['acc_own_set'] = kernel.accuracies_own_set
            data_to_save[kernel.name]['processing_time'] = kernel.processing_time

        # Save the data
        filename = 'thresholding'
        save_results(data_to_save, filename)

    def plot_results(self, save=False):
        fig_size = (6.4, 3.5)
        plt.figure(figsize=fig_size)
        plt.xlabel('Threshold value', fontsize=14)
        plt.ylabel('Accuracy [%]')

        for kernel in self.kernels:
            accuracy_values = kernel.accuracies_own_set
            threshold_values = self.threshold_values

            plt.plot(threshold_values, accuracy_values, label=kernel.name)

        plt.xlim([0, 1])
        plt.grid(which='both', linestyle='dashed')
        plt.legend(loc='best')
        plt.title("Accuracy vs Threshold value", fontsize=14)
        plt.tight_layout()
        if save:
            filename = 'thresholding'
            path = 'figures/' + filename + '.eps'
            if os.path.exists(path):
                os.remove(path)
            plt.savefig(path, format='eps')
        plt.show()


# Launch the test
test_preprocessing = TestPreprocessing()
