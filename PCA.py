
# Principal Component Analysis on training and test data sets
import time
import joblib
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from real_dataset import RealDataset
from tools import *


class Kernel:
    def __init__(self, name, n_samples, threshold):
        self.name = name
        self.n_samples = n_samples
        self.threshold = threshold
        self.number_of_principal_components = [3, 5, 10, 15, 20, 30, 40, 50]

        # Data to fill
        self.accuracies_training_set = [None] * len(self.number_of_principal_components)
        self.accuracies_test_set = [None] * len(self.number_of_principal_components)
        self.accuracies_own_set = [None] * len(self.number_of_principal_components)
        self.processing_time = None


class TestPCA:
    def __init__(self):
        kernels = list()
        kernels.append(Kernel('LinearSVC', 10000, 0.35))
        kernels.append(Kernel('linear', 10000, 0.1))
        kernels.append(Kernel('poly', 10000, 0.1))
        kernels.append(Kernel('rbf', 10000, 0.1))
        kernels.append(Kernel('sigmoid', 10000, 0.1))
        self.kernels = kernels

        # Launch test
        self.test()

        # Print results
        self.print_info()

    def test(self):
        for kernel, i in zip(self.kernels, range(len(self.kernels))):
            # Importing MNIST dataset
            dataset = import_mnist_dataset(kernel.n_samples, True)

            # Splitting dataset
            dataset.separate_train_test(0.2)

            # Own designed dataset
            real_dataset = RealDataset(n_images=80, threshold=kernel.threshold)

            # Start counting time
            t0 = time.process_time()

            for number_pc, j in zip(kernel.number_of_principal_components,
                                    range(len(kernel.number_of_principal_components))):
                # Creating classifier
                if kernel.name == 'LinearSVC':
                    clf = svm.LinearSVC()
                else:
                    clf = svm.SVC(kernel=kernel.name)

                # PCA decomposition
                pca = PCA(n_components=number_pc)
                X_train = pca.fit_transform(dataset.X_train)
                X_test = pca.transform(dataset.X_test)
                X_real = pca.transform(real_dataset.X)

                # Fitting classifier
                clf.fit(X=X_train, y=dataset.y_train)

                # Saving scores
                acc_training_set = clf.score(X_train, dataset.y_train)
                acc_test_set = clf.score(X_test, dataset.y_test)
                acc_real_set = clf.score(X_real, real_dataset.y)
                print('acc:', acc_real_set)
                self.kernels[i].accuracies_training_set[j] = acc_training_set
                self.kernels[i].accuracies_test_set[j] = acc_test_set
                self.kernels[i].accuracies_own_set[j] = acc_real_set

            # Processing time
            processing_time = time.process_time() - t0
            self.kernels[i].processing_time = processing_time

    def print_info(self):
        for kernel in self.kernels:
            print(100*'-')
            print(f"SVC classifier trained: {kernel.name}")

            print(f"Number of Principal Components: {kernel.number_of_principal_components}")
            print(f"Training set scores: {kernel.accuracies_training_set}")
            print(f"Test set scores: {kernel.accuracies_test_set}")
            print(f"Real set scores: {kernel.accuracies_own_set}")

            print(f"Processing time: {kernel.processing_time}")


principal_components_test = TestPCA()

# Import dataset
# n_samples = 20_000
# dataset = import_mnist_dataset(n_samples, True)
#
# # Dataset preparation
# # Split the MNIST dataset in training and testing sets
# print("Separating dataset in training and testing sets ...")
# test_ratio = 0.2
# dataset.separate_train_test(test_ratio)
# print(f"Dataset separated, we have a {100 * test_ratio} % of samples for the test set!")
# Scale
# dataset.scaler_fit_transform(dataset.X_train)
# dataset.scaler_fit(dataset.X_test)
# dataset.dump_scaler()
# PCA
# n_components = 50
# pca = PCA(n_components=n_components)
# X_train = pca.fit_transform(dataset.X_train)
# X_test = pca.transform(dataset.X_test)

# # Plotting the data
# y_train = dataset.y_train
# u_labels = np.unique(y_train)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# for i in u_labels:
#     ax.scatter(X_train[y_train == i][:, 0], X_train[y_train == i][:, 1], X_train[y_train == i][:, 2], label=i)
#
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
# plt.show()
#
# print("Starting training...")
# t = time.process_time()
# clf = svm.LinearSVC()
# clf.fit(X_train, dataset.y_train)
#
# print('Processing time : ' + str(round((time.process_time() - t), 2)))
# print('Accuracy on training set = ' + str(round(clf.score(X_train, dataset.y_train), 2)))
# print('Accuracy on testing set = ' + str(round(clf.score(X_test, dataset.y_test), 2)))

# Grid search to find the best LinearSVC() classifier
# Define classifier
# clf_search = svm.LinearSVC()
#
# # Define search space
# print("Defining the search space...")
# space = dict()
# space['C'] = [10 ** x for x in range(-3, 3)]
# space['dual'] = [True, False]
# # space['gamma'] = [10 ** x for x in range(-3, 3)]
# print("Search space defined!")
#
# # Define search
# print("Defining search...")
# t = time.process_time()
# search = GridSearchCV(clf_search, space, n_jobs=-1, verbose=3)
# print("Search defined!")
#
# # Execute search
# print("Executing search...")
# t = time.process_time()
# clf_result = search.fit(dataset.X_train, dataset.y_train)
# print("Search executed, best solution found!")
#
# print(f"LinearSVC() classifier.")
# # print(f"The dataset has been decomposed in {n_components} Principal Components.")
# # print(f"Variance ratio : {100*round(np.sum(pca.explained_variance_ratio_), 4)} %")
#
# # Summarize result
# print(f'Time spent in search : {time.process_time() - t}')
# print('Best Score: %s' % clf_result.best_score_)
# print('Best Hyperparameters: %s' % clf_result.best_params_)
#
# # Testing on test set
# print('Accuracy on training set: ' + str(round(clf_result.best_estimator_.score(dataset.X_train, dataset.y_train), 2)))
# print('Accuracy on testing set: ' + str(round(clf_result.best_estimator_.score(dataset.X_test, dataset.y_test), 2)))
# print('Accuracy on testing set: ' + str(round(clf_result.best_estimator_.score(dataset.X_test, dataset.y_test), 2)))
#
# # Testing on real images
# real_set = RealDataset()
# X_real_test = dataset.scaler.transform(real_set.X)
# # X_real_test = pca.transform(X_real_test)
# print('Accuracy on real images: ' + str(round(clf_result.best_estimator_.score(X_real_test, real_set.y), 2)))
















