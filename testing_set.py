from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from skimage.transform import resize
from skimage import io, exposure
import numpy as np
import matplotlib.pyplot as plt
import pickle


def import_images(nb_images):
    data = dict()
    data['data'] = list()
    data['target'] = list()
    for i in range(nb_images):
        path = 'RealSet/' + str(i) + '.png'

        # Loading the image
        image = io.imread(path, as_gray=True)

        # Changing the size to meet the classifier number of parameters
        image = resize(image, (28, 28), anti_aliasing=False)
        image = image

        # Mapping the image from 0 to 1 with a lineal function
        min = image.min()
        max = image.max()
        m = 1 / (max - min)

        # y = m*x + b, b = -m*min
        # image = m * image - m * min
        image = exposure.rescale_intensity(image)
        # percentiles = np.percentile(image, (0.5, 99.5))
        # image = exposure.rescale_intensity(image, in_range=tuple(percentiles))

        # Thresholding the image to accentuate the white background
        threshold = 0.2
        # image = np.where(image > threshold, image, 0)
        # show_image(image)
        # Transform the matrix(image) into a 1d vector
        image = image.reshape(1, 784)
        if i == 1:
            show_image(image)

        # Storing the vectors and targets
        data['data'].append(image)
        data['targets'] .append(str(i % 10))
    return data


def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    import itertools
    import matplotlib.pyplot as plt
    import numpy as np

    accuracy = np.trace(cm) / float(np.sum(cm))
    mis_class = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label -> Accuracy={:0.4f}, Misclass={:0.4f}'.format(accuracy, mis_class))
    plt.show()


def predict_with_specified_classifier(filename, data, targets):
    # Loading the trained classifier
    clf = pickle.load(open(filename, 'rb'))

    # Making the predictions
    predictions = []
    for i in range(len(data)):
        predictions.append(clf.predict(data[i]))

    # Accuracy on the real test set
    print('Accuracy in real test set:', accuracy_score(targets, predictions))

    # Showing confusion matrix
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # titles_options = [("Confusion matrix, without normalization", False), ("Normalized confusion matrix", True)]
    titles_options = [("Confusion matrix, without normalization", False)]

    conf_mx = confusion_matrix(targets, predictions)
    plt.figure(figsize=(10, 6))

    for title, normalize in titles_options:
        plot_confusion_matrix(cm=conf_mx, target_names=class_names, title=title, cmap=plt.cm.Greens,
                              normalize=normalize)

    print(targets)
    print(predictions)


def show_image(image):
    image = image.reshape(28, 28)
    plt.figure()
    plt.gray()
    plt.matshow(image)
    plt.show()


# Custom images
# Loading the images set
number_images = 40
data, targets = import_images(number_images)

# Testing with LinearSVC() classifier
filename = 'LinearSVC.pkl'
predict_with_specified_classifier(filename, data, targets)

# # Loading the trained classifiers
# kernels = ['linear', 'poly', 'rbf', 'sigmoid']
# # kernels = ['linear']
# for kernel in kernels:
#     print('kernel: ', kernel)
#     filename = 'svm_kernel_' + kernel + '_wb.pkl'
#     predict_with_specified_classifier(filename, data, targets)
