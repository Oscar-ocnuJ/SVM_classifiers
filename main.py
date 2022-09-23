# Module importation
import time

from tools import save_classifier, train_classifier, import_mnist_dataset
from dataset import Dataset
from sklearn import svm
import joblib
from tools import *
from real_dataset import RealDataset
import matplotlib.pyplot as plt


performance_vs_samples = {'kernel': 'Linear SVC',
                          'dataset_sizes': list(),
                          'train_set_acc': list(),
                          'test_set_acc': list(),
                          'real_set_acc': list(),
                          'processing_times': list(),
                          'total_processing_time': list()}

dataset_sizes = list(range(100, 901, 100))
dataset_sizes.extend(list(range(1_000, 9_001, 1_000)))
dataset_sizes.extend(list(range(10_000, 20_001, 2_000)))

performance_vs_samples['dataset_sizes'] = dataset_sizes

t0 = time.process_time()
for size in performance_vs_samples['dataset_sizes']:
    # Load the MNIST dataset in Dataset class and analyse it
    print(100*'-')
    n_samples = size
    dataset = import_mnist_dataset(n_samples)

    # Dataset preparation
    # Split the MNIST dataset in training and testing sets
    print("Separating dataset in training and testing sets ...")
    test_ratio = 0.2
    dataset.separate_train_test(test_ratio)
    print("Dataset separated !")

    # LinearSVC classifier
    # print('Training a linear SVM classifier, one-vs-the-rest approach...')
    print('Training a SVC ' + performance_vs_samples['kernel'] + ' classifier... ')
    clf = svm.LinearSVC()

    t_clf = time.process_time()
    clf.fit(dataset.X_train, dataset.y_train)
    accuracy_train_set = round(clf.score(dataset.X_train, dataset.y_train), 2)
    accuracy_test_set = round(clf.score(dataset.X_test, dataset.y_test), 2)
    processing_time = round((time.process_time() - t_clf), 4)
    print('Processing time: ' + str(processing_time) + ' s')
    print('Accuracy on training set: ' + str(accuracy_train_set))
    print('Accuracy on testing set: ' + str(accuracy_test_set))

    # Testing in the real dataset
    real_dataset = RealDataset()
    accuracy_real_set = round(clf.score(real_dataset.X, real_dataset.y), 2)
    print('Accuracy on real test set: ' + str(accuracy_real_set))

    performance_vs_samples['train_set_acc'].append(accuracy_train_set)
    performance_vs_samples['test_set_acc'].append(accuracy_test_set)
    performance_vs_samples['real_set_acc'].append(accuracy_real_set)
    performance_vs_samples['processing_times'].append(processing_time)

total_processing_time = round((time.process_time() - t0), 4)
print('Total processing time: ' + str(total_processing_time) + ' s')

filename = 'results/' + 'performance_vs_samples_' + performance_vs_samples['kernel'] + '.dict_pkl'
pickle.dump(performance_vs_samples, open(filename, 'wb'))

# Plotting the performance
plt.figure()
fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('Number of samples', fontsize=12)
ax1.set_ylabel('Accuracy', color=color)
ax1.semilogx(performance_vs_samples['dataset_sizes'], performance_vs_samples['train_set_acc'], '-bo',
             label='Training set')
ax1.semilogx(performance_vs_samples['dataset_sizes'], performance_vs_samples['test_set_acc'], '--b^', label='Test set')
ax1.semilogx(performance_vs_samples['dataset_sizes'], performance_vs_samples['real_set_acc'], '-.bs', label='Own set')
ax1.tick_params(axis='y', labelcolor=color)
plt.grid(which='both', linestyle='dashed')
plt.legend(loc='best')
plt.title("Performance of " + performance_vs_samples['kernel'] + " classifier", fontsize=14)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Time [s]', color=color)
ax2.plot(performance_vs_samples['dataset_sizes'], performance_vs_samples['processing_times'], '-r>',
         label='Processing time')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
filename = 'performance_svc_' + performance_vs_samples['kernel']
path = 'figures/' + filename + '.eps'
plt.savefig(path, format='eps')
plt.show()

