import os
import pickle
import joblib
from sklearn.datasets import fetch_openml
from dataset import Dataset


# Python script with all the required functions
def import_mnist_dataset(n_samples, transform=False):
    """
    Import n_samples from MNIST dataset directly from the datasets' folder, if it is not found, it will be downloaded
    from the official website. The original MNIST dataset has 70_000 samples
    :param n_samples: The number of samples to create a subset from MNIST
    :param transform: Control parameter to enable the transformation in the set
    :return: A 'Dataset' object whit. See 'Dataset' class documentation
    """
    mnist_path = r'datasets/mnist_784_full.dataset_job'
    if os.path.exists(mnist_path):
        print("Importing MNIST dataset from a local directory...")
        mnist = joblib.load(mnist_path)
    else:
        print("Importing MNIST dataset from their website...")
        mnist = fetch_openml('mnist_784')
    print("MNIST dataset imported with success!")

    print('Choosing a fraction of the MNIST dataset to train and test the classifier...')
    dataset = Dataset(mnist, n_samples, transform)
    print(f'Dataset of {n_samples} samples created from MNIST!')

    return dataset


def save_classifier(classifier, filename):
    """
    This function uses 'pickle (Python Object Serialization)' module to save a classifier.
    If a file with the same name already exists, it will be deleted and the new file saved.
    The extension of the generated file is '.model_pkl'
    :param classifier: The classifier to be saved
    :param filename: The filename without an extension
    :return: The object stored in the file
    """
    file_path = 'classifiers/' + filename + '.model_pkl'
    if os.path.exists(file_path):
        os.remove(file_path)
    pickle.dump(classifier, open(file_path, 'wb'))


def save_results(results, filename):
    file_path = 'results/' + filename + '.results_pkl'
    if os.path.exists(file_path):
        os.remove(file_path)
    pickle.dump(results, open(file_path, 'wb'))


def import_results(filename):
    file_path = 'results/' + filename + '.results_pkl'
    if not os.path.exists(file_path):
        return None
    return pickle.load(open(file_path, 'rb'))


def save_search_results(search_result, filename):
    """
    Save a result from a random or grid search of parameters
    :param search_result: Result of a search process
    :param filename: The filename to be saved
    :return: The results saved in the results' folder.
    """
    file_path = 'results/' + filename + '.cv_results_pkl'
    if os.path.exists(file_path):
        os.remove(file_path)
    pickle.dump(search_result, open(file_path, 'wb'))


def import_search_results(filename):
    file_path = 'results/' + filename + '.cv_results_pkl'
    if not os.path.exists(file_path):
        return None
    else:
        search_results = pickle.load(open(file_path, 'rb'))

    return search_results


def import_classifier(filename):
    """
    Use pickle to import a previously saved classifier also with pickle module. If the file does not exist, the
    function return is None
    :param filename: 'str' indicating the name of the classifier to be imported
    :return: A svm classifier.
    """
    file_path = 'classifiers/' + filename + '.model_pkl'
    if not os.path.exists(file_path):
        return None
    else:
        clf = pickle.load(open(file_path, 'rb'))

    return clf
