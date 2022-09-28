# Random Search SVM classifier model on digit recognition
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform
import time
from sklearn import svm
from tools import *
import matplotlib.pyplot as plt
from glob import glob


class Kernel:
    def __init__(self, name):
        self.name = name
        self.result = [None] * 2
        self.processing_time = [None] * 2


class TestRandomizedSearchCV:
    def __init__(self):
        # Define search space
        space = dict()
        space['C'] = loguniform(1e-5, 1e5)
        space['gamma'] = loguniform(1e-15, 1e5)

        self.space = space
        self.n_iterations = 1_000

        kernels = list()
        kernels.append(Kernel('LinearSVC'))
        kernels.append(Kernel('linear'))
        kernels.append(Kernel('poly'))
        kernels.append(Kernel('rbf'))
        kernels.append(Kernel('sigmoid'))
        self.kernels = kernels

        # Fill object
        base_name = 'randomized_search_cv_SVC_*'
        if glob('results/' + base_name):
            # Load the previous test
            for kernel, i in zip(kernels, range(len(kernels))):
                self.kernels[i].result[0] = import_results(base_name.strip('*') + kernel.name)

            kernels[3].result[1] = import_results(base_name.strip('*') + 'rbf_gamma')

        else:
            # Launch search
            self.search()

    def search(self, n_samples_in_dataset=500, test_ratio=0.2):
        # Loading a subset from MNIST dataset
        dataset = import_mnist_dataset(n_samples_in_dataset)

        # Dataset preparation
        # Split the MNIST dataset in training and testing sets
        print("Separating dataset in training and testing sets ...")
        dataset.separate_train_test(test_ratio)
        print(f"Dataset separated, we have a {100 * test_ratio} % of samples for the test set!")

        print("Randomized search for 'C' parameter...")
        space = {'C': self.space['C']}
        for kernel, i in zip(self.kernels, range(len(self.kernels))):

            if kernel.name == 'LinearSVC':
                clf = svm.LinearSVC()
            else:
                clf = svm.SVC(kernel=kernel.name)

            # Define search
            search = RandomizedSearchCV(clf, space, n_iter=self.n_iterations, n_jobs=-1, random_state=1, verbose=3)

            # Execute search
            print(100 * '-')
            print(f"Randomized Search CV in a SVC {kernel.name} classifier!")
            print("Executing search...")
            t = time.process_time()
            result = search.fit(dataset.X_train, dataset.y_train)
            filename = 'randomized_search_cv_SVC_' + kernel.name

            # Saving results
            save_results(result, filename)
            self.kernels[i].result[0] = result
            print("Randomized search executed, best solution found!")

            # Summarize result
            processing_time = time.process_time() - t
            self.kernels[i].processing_time[0] = processing_time
            print(f'Time spent in {self.n_iterations} iterations of search : {processing_time}')
            print('Best Score: %s' % result.best_score_)
            print('Best Hyperparameters: %s' % result.best_params_)

        print("Randomized search for 'gamma' parameter in SVC 'rbf' classifier")
        space = {'gamma': self.space['gamma']}
        clf = svm.SVC(kernel='rbf')

        # Define search
        search = RandomizedSearchCV(clf, space, n_iter=self.n_iterations, n_jobs=-1, random_state=1, verbose=3)

        print(100 * '-')
        print(f"Randomized Search CV in a SVC 'rbf' classifier!")
        # Execute search
        print("Executing search...")
        t = time.process_time()
        result = search.fit(dataset.X_train, dataset.y_train)
        filename = 'randomized_search_cv_SVC_' + 'rbf_gamma'
        save_results(result, filename)
        self.kernels[3].result[1] = result
        print("Randomized search executed, best solution found!")

        # Summarize result
        processing_time = time.process_time() - t
        self.kernels[3].processing_time[1] = processing_time
        print(f'Time spent in {self.n_iterations} iterations of search : {processing_time}')
        print('Best Score: %s' % result.best_score_)
        print('Best Hyperparameters: %s' % result.best_params_)

    def plot_results(self, save=False):
        fig_size = (6.4, 3.5)
        plt.figure(figsize=fig_size)
        plt.xlabel('C', fontsize=14)
        plt.ylabel('Accuracy [%]')

        for kernel in self.kernels:
            cv_results_ = kernel.result[0].cv_results_
            c_values = cv_results_['param_C']
            score_values = 100 * cv_results_['mean_test_score']
            c_values, score_values = (list(t) for t in zip(*sorted(zip(c_values, score_values))))

            plt.semilogx(c_values, score_values, label=kernel.name)

        plt.grid(which='both', linestyle='dashed')
        plt.legend(loc='best')
        plt.title("Randomized search for SVC classifiers - 'C'", fontsize=14)
        plt.tight_layout()
        if save:
            filename = 'randomized_search_vs_C'
            path = 'figures/' + filename + '.eps'
            if os.path.exists(path):
                os.remove(path)
            plt.savefig(path, format='eps')
        plt.show()

        plt.figure(figsize=fig_size)
        plt.xlabel('gamma', fontsize=14)
        plt.ylabel('Accuracy [%]')

        kernel = self.kernels[3]
        result = kernel.result[1].cv_results_
        gamma_values = result['param_gamma']
        score_values = 100 * result['mean_test_score']
        gamma_values, score_values = (list(t) for t in zip(*sorted(zip(gamma_values, score_values))))
        plt.semilogx(gamma_values, score_values, label=kernel.name)
        plt.grid(which='both', linestyle='dashed')
        plt.legend(loc='best')
        plt.title("Randomized search for SVC classifiers - 'gamma'", fontsize=14)
        plt.tight_layout()
        if save:
            filename = 'randomized_search_vs_gamma'
            path = 'figures/' + filename + '.eps'
            if os.path.exists(path):
                os.remove(path)
            plt.savefig(path, format='eps')
        plt.show()


# Executing search
randomized_search_cv = TestRandomizedSearchCV()
# Plot search
randomized_search_cv.plot_results(save=True)
