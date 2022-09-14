# Module importation
from tools import save_classifier, train_classifier, import_mnist_dataset
from dataset import Dataset
from sklearn import svm
import joblib
from tools import *
from real_dataset import RealDataset


# Load the MNIST dataset in Dataset class and analyse it
n_samples = 20_000
dataset = import_mnist_dataset(n_samples)

# Display some digits with their corresponding features
samples_to_display = 5
for i in range(samples_to_display):
    dataset.digits[i].print()
print(f'{samples_to_display} samples displayed!')

# Display digits repartition
dataset.print_info()
print('Digits repartition displayed!')

# Dataset preparation
# Split the MNIST dataset in training and testing sets
print("Separating dataset in training and testing sets ...")
test_ratio = 0.2
dataset.separate_train_test(test_ratio)
print("Dataset separated !")

# Display the repartition of the digits
dataset.display_train_test()
print('Repartition of sets displayed!')

# LinearSVC classifier
# print('Training a linear SVM classifier, one-vs-the-rest approach...')
print('Training a SVC kernel rbf classifier... ')
clf = svm.SVC(kernel='rbf')

t = time.process_time()
# dataset.scaler_fit_transform(dataset.X_train)
clf.fit(dataset.X_train, dataset.y_train)
print('Processing time: ' + str(round((time.process_time() - t), 2)) + ' s')
print('Accuracy on training set: ' + str(round(clf.score(dataset.X_train, dataset.y_train), 2)))
print('Accuracy on testing set: ' + str(round(clf.score(dataset.X_test, dataset.y_test), 2)))

# Testing in the real dataset
real_dataset = RealDataset()
X_real = dataset.scaler.transform(real_dataset.X)
print('Accuracy on real test set: ' + str(round(clf.score(X_real, real_dataset.y), 2)))


# # SVM classifier with different kernels
# print('Beginning the training of the classifiers, testing different kernels for SCV()...')
# kernels = ['linear', 'poly', 'rbf', 'sigmoid']
# for ker in kernels:
#     print('Kernel: ' + ker)
#     clf_type = 'SVC'
#     filename = clf_type + '_kernel_' + ker
#     clf = svm.SVC(kernel=ker)
#     save_classifier(clf, filename)
#     train_classifier(clf, dataset)
# print('Classifiers trained and saved!')
#
# # NuSVC classifier with different kernels
# print('Beginning the training of the classifiers, testing different kernels for NuSVC()...')
# kernels = ['linear', 'poly', 'rbf', 'sigmoid']
# for ker in kernels:
#     print('Kernel: ' + ker)
#     clf_type = 'NuSVC'
#     filename = clf_type + '_kernel_' + ker
#     clf = svm.NuSVC(kernel=ker)
#     save_classifier(clf, filename)
#     train_classifier(clf, dataset)
