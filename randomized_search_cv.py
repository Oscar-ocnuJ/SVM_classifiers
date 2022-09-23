# Random Search SVM classifier model on digit recognition
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform
import time
from dataset import Dataset
from sklearn import svm
import pandas as pd
from tools import *
from real_dataset import RealDataset
import matplotlib.pyplot as plt


class Kernel:
    def __int__(self, name):
        self.name = name
        self.result = None
        self.processing_time = 0


class TestRandomizedSearchCV:
    def __init__(self, space, n_iterations=10_000):
        self.space = space
        self.n_iterations = n_iterations

        kernels = list()
        kernels.append(Kernel('LinearSVC'))
        kernels.append(Kernel('linear'))
        kernels.append(Kernel('poly'))
        kernels.append(Kernel('rbf'))
        kernels.append(Kernel('sigmoid'))
        self.kernels = kernels

    def search(self, n_samples_in_dataset=500, test_ratio=0.2):
        # Loading a subset from MNIST dataset
        dataset = import_mnist_dataset(n_samples_in_dataset)

        # Dataset preparation
        # Split the MNIST dataset in training and testing sets
        print("Separating dataset in training and testing sets ...")
        dataset.separate_train_test(test_ratio)
        print(f"Dataset separated, we have a {100*test_ratio} % of samples for the test set!")

        for kernel, i in zip(self.kernels, range(len(self.kernels))):

            if kernel.name == 'LinearSVC':
                clf = svm.LinearSVC()
            else:
                clf = svm.SVC(kernel=kernel.name)

            # Define search
            search = RandomizedSearchCV(clf, self.space, n_iter=self.n_iterations, n_jobs=-1, random_state=1, verbose=3)

            # Execute search
            print("Executing search...")
            t = time.process_time()
            result = search.fit(dataset.X_train, dataset.y_train)
            self.kernels[i].result = result
            print("Randomized search executed, best solution found!")

            # Summarize result
            print(f'Time spent in {n_iterations} iterations of search : {time.process_time() - t}')
            print('Best Score: %s' % result.best_score_)
            print('Best Hyperparameters: %s' % result.best_params_)




# Define search space
print("Defining the search space...")
space = dict()
# space['kernel'] = ['sigmoid']
space['C'] = loguniform(1e-5, 1e5)
# space['gamma'] = loguniform(1e-15, 1e5)
print("Search space defined!")

# Define search
print("Defining search...")
n_iterations = 10000
search = RandomizedSearchCV(clf, space, n_iter=n_iterations, scoring='accuracy', n_jobs=-1, random_state=1, verbose=3)
print("Search defined!")

# Execute search
print("Executing search...")
t = time.process_time()
result = search.fit(dataset.X_train, dataset.y_train)



# Results in test set
clf = result.best_estimator_
print(f"Score in test set: {clf.score(dataset.X_test, dataset.y_test)}")

# Results in real images set
real_dataset = RealDataset()
print(f"Score in real set: {clf.score(real_dataset.X, real_dataset.y)}")

# Saving the search results
filename = 'randomized_search_cv_SVC_' + 'LinearSVC'
save_search_results(result, filename)



