# Random Search SVM classifier model on digit recognition
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform
import time
from dataset import Dataset
from sklearn import svm
import pandas as pd

# Load dataset
print('Importing MNIST dataset ...')
filename = 'mnist_784.ob'
mnist = joblib.load(filename)
print('MNIST dataset imported with success!')

# Loading a subset from MNIST dataset
samples = 20_000
print('Creating a subset and giving the desired structure...')
training_set = Dataset(mnist, samples)
print('Set created from MNIST!')

# Dataset preparation
# Split the MNIST dataset in training and testing sets
print("Separating dataset in training and testing sets ...")
test_ratio = 0.2
training_set.separate_train_test(test_ratio)
print(f"Dataset separated, we have a {100*test_ratio} % of samples for the test set!")

# Define classifier model
clf = svm.SVC()

# Define search space
print("Defining the search space...")
space = dict()
space['kernel'] = ['linear', 'poly', 'rbf']
space['C'] = loguniform(1e-2, 100)
space['gamma'] = ['scale', 'auto']
print("Search space defined!")

# Define search
print("Defining search...")
n_iterations = 10
search = RandomizedSearchCV(clf, space, n_iter=n_iterations, scoring='accuracy', n_jobs=3, random_state=1)
print("Search defined!")

# Execute search
print("Executing search...")
t = time.process_time()
result = search.fit(training_set.X_train, training_set.y_train)
print("Search executed, best solution found!")

# Summarize result
print(f'Time spent in {n_iterations} iterations of search : {time.process_time() - t}')
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# Saving the best found model
filename = 'best_found_clf.model'
joblib.dump(result, filename)

# Saving the cv_results_ as csv file
df = pd.DataFrame(result.cv_results_)
df.to_csv('randomized_search_cv_results.csv')


