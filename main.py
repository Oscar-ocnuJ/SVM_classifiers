# Module importation
from sklearn import svm
from tools import *
from real_dataset import RealDataset
import matplotlib.pyplot as plt


class Kernel:
    def __init__(self, name, length):
        self.name = name
        self.train_set_accuracies = [None]*length
        self.test_set_accuracies = [None]*length
        self.real_set_accuracies = [None]*length
        self.processing_times = [None]*length
        self.total_processing_time = 0


class TestPerformanceVsSamplesNumber:
    def __init__(self):
        dataset_sizes = list(range(100, 901, 100))
        dataset_sizes.extend(list(range(1_000, 9_001, 1_000)))
        dataset_sizes.extend(list(range(10_000, 20_001, 2_000)))
        self.dataset_sizes = dataset_sizes

        kernels = list()
        length = len(self.dataset_sizes)
        kernels.append(Kernel('LinearSVC', length))
        kernels.append(Kernel('linear', length))
        kernels.append(Kernel('poly', length))
        kernels.append(Kernel('rbf', length))
        kernels.append(Kernel('sigmoid', length))
        self.kernels = kernels

        # Launch test
        self.test()

    def test(self):
        real_dataset = RealDataset()
        for kernel, i in zip(self.kernels, range(len(self.kernels))):
            t0 = time.process_time()
            for n_samples, j in zip(self.dataset_sizes, range(len(self.dataset_sizes))):
                # Load the MNIST dataset in Dataset class and analyse it
                print(100*'-')
                dataset = import_mnist_dataset(n_samples)

                # Dataset preparation
                # Split the MNIST dataset in training and testing sets
                print("Separating dataset in training and testing sets ...")
                test_ratio = 0.2
                dataset.separate_train_test(test_ratio)
                print("Dataset separated!")

                # LinearSVC classifier
                # print('Training a linear SVM classifier, one-vs-the-rest approach...')
                print('Training a SVC ' + kernel.name + ' classifier... ')
                if kernel.name == 'LinearSVC':
                    clf = svm.LinearSVC()
                else:
                    clf = svm.SVC(kernel=kernel.name)

                t_clf = time.process_time()
                clf.fit(dataset.X_train, dataset.y_train)
                accuracy_train_set = 100*round(clf.score(dataset.X_train, dataset.y_train), 4)
                accuracy_test_set = 100*round(clf.score(dataset.X_test, dataset.y_test), 4)
                processing_time = round((time.process_time() - t_clf), 4)
                print('Processing time: ' + str(processing_time) + ' s')
                print('Accuracy on training set: ' + str(accuracy_train_set) + ' %')
                print('Accuracy on testing set: ' + str(accuracy_test_set) + ' %')

                # Testing in the real dataset
                accuracy_real_set = 100*round(clf.score(real_dataset.X, real_dataset.y), 4)
                print('Accuracy on real test set: ' + str(accuracy_real_set) + ' %')

                self.kernels[i].train_set_accuracies[j] = accuracy_train_set
                self.kernels[i].test_set_accuracies[j] = accuracy_test_set
                self.kernels[i].real_set_accuracies[j] = real_dataset
                self.kernels[i].processing_times[j] = processing_time

            total_processing_time = round((time.process_time() - t0), 4)
            self.kernels[i].total_processing_time = total_processing_time
            print('Total processing time: ' + str(total_processing_time) + ' s')

    def plot_graphs(self, save=False):
        for kernel in self.kernels:
            # Plotting the performance
            plt.figure()
            fig, ax1 = plt.subplots()
            color = 'tab:blue'
            ax1.set_xlabel('Number of samples in the dataset', fontsize=12)
            ax1.set_ylabel('Accuracy [%]', color=color)
            ax1.semilogx(self.dataset_sizes, kernel.train_set_accuracies, '-bo', label='Training set')
            ax1.semilogx(self.dataset_sizes, kernel.test_set_accuracies, '--b^', label='Test set')
            ax1.tick_params(axis='y', labelcolor=color)
            plt.grid(which='both', linestyle='dashed')
            plt.legend(loc='best')
            plt.title("Performance of " + kernel.name + " classifier", fontsize=14)

            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Time [s]', color=color)
            ax2.plot(self.dataset_sizes, kernel.processing_times, '-r>', label='Processing time')
            ax2.tick_params(axis='y', labelcolor=color)

            fig.tight_layout()
            if save:
                filename = 'performance_svc_' + kernel.name
                path = 'figures/' + filename + '.eps'
                plt.savefig(path, format='eps')
            plt.show()


# Launch the test
performance_test = TestPerformanceVsSamplesNumber()
save_results(performance_test, 'performance_vs_samples')

# Plot performance
performance_test.plot_graphs()
