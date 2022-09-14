# Random Search SVM classifier model on digit recognition
from sklearn.model_selection import GridSearchCV
import time
from sklearn import svm
from tools import import_mnist_dataset
from real_dataset import RealDataset

# Load the MNIST set
dataset = import_mnist_dataset(20_000, True)

# Dataset preparation
# Split the MNIST dataset in training and testing sets
print("Separating dataset in training and testing sets ...")
test_ratio = 0.2
dataset.separate_train_test(test_ratio)
print(f"Dataset separated, we have a {100 * test_ratio} % of samples for the test set!")

# Define classifier model
clf = svm.LinearSVC()

# Define search space
print("Defining the search space...")
space = dict()
space['C'] = [2 ** x for x in range(-1, 2)]
# space['gamma'] = [2 ** x for x in range(-1, 2)]
space['dual'] = [True, False]
print("Search space defined!")

# Define search
print("Defining search...")
search = GridSearchCV(clf, space, n_jobs=-1, verbose=3)
print("Search defined!")

# Execute search
print("Executing search...")
t = time.process_time()
result = search.fit(dataset.X_train, dataset.y_train)
print("Search executed, best solution found!")

# Summarize result
print(f'Time spent in search : {time.process_time() - t}')
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# Results on the sets
print(f"Accuracy in training set:{result.best_estimator_.score(dataset.X_train, dataset.y_train)}")
print(f"Accuracy in testing set:{result.best_estimator_.score(dataset.X_test, dataset.y_test)}")
# Testing the real test set
real_dataset = RealDataset()


# # Saving the best found model
# filename = 'best_rbf_clf.model_job'
# joblib.dump(result, filename)
#
# # Saving the cv_results_ as csv file
# df = pd.DataFrame(result.cv_results_)
# df.to_csv('grid_search_cv_results.csv')
