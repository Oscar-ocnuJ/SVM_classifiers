
# Principal Component Analysis on training and test data sets
import time
import joblib
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sklearn.model_selection import GridSearchCV
from real_dataset import RealDataset
from tools import *

# Import the 20k samples dataset
n_samples = 20_000
dataset = import_mnist_dataset(n_samples, True)

# Dataset preparation
# Split the MNIST dataset in training and testing sets
print("Separating dataset in training and testing sets ...")
test_ratio = 0.2
dataset.separate_train_test(test_ratio)
print(f"Dataset separated, we have a {100 * test_ratio} % of samples for the test set!")
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
clf_search = svm.LinearSVC()

# Define search space
print("Defining the search space...")
space = dict()
space['C'] = [10 ** x for x in range(-3, 3)]
space['dual'] = [True, False]
# space['gamma'] = [10 ** x for x in range(-3, 3)]
print("Search space defined!")

# Define search
print("Defining search...")
t = time.process_time()
search = GridSearchCV(clf_search, space, n_jobs=-1, verbose=3)
print("Search defined!")

# Execute search
print("Executing search...")
t = time.process_time()
clf_result = search.fit(dataset.X_train, dataset.y_train)
print("Search executed, best solution found!")

print(f"LinearSVC() classifier.")
# print(f"The dataset has been decomposed in {n_components} Principal Components.")
# print(f"Variance ratio : {100*round(np.sum(pca.explained_variance_ratio_), 4)} %")

# Summarize result
print(f'Time spent in search : {time.process_time() - t}')
print('Best Score: %s' % clf_result.best_score_)
print('Best Hyperparameters: %s' % clf_result.best_params_)

# Testing on test set
print('Accuracy on training set: ' + str(round(clf_result.best_estimator_.score(dataset.X_train, dataset.y_train), 2)))
print('Accuracy on testing set: ' + str(round(clf_result.best_estimator_.score(dataset.X_test, dataset.y_test), 2)))
print('Accuracy on testing set: ' + str(round(clf_result.best_estimator_.score(dataset.X_test, dataset.y_test), 2)))

# Testing on real images
real_set = RealDataset()
X_real_test = dataset.scaler.transform(real_set.X)
# X_real_test = pca.transform(X_real_test)
print('Accuracy on real images: ' + str(round(clf_result.best_estimator_.score(X_real_test, real_set.y), 2)))
















