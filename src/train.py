import argparse
import os
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression

from azureml.core import Run
from utils.mnist_utils import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--regularization', type=float, dest='reg', default=0.01, help='regularization rate')


def train(data_folder, regularization, output):

    # load train and test set into numpy arrays
    # note we scale the pixel intensity values to 0-1 (by dividing it with 255.0) so the model can converge faster.
    X_train = load_data(os.path.join(data_folder, 'train-images.gz'), False) / 255.0
    X_test = load_data(os.path.join(data_folder, 'test-images.gz'), False) / 255.0
    y_train = load_data(os.path.join(data_folder, 'train-labels.gz'), True).reshape(-1)
    y_test = load_data(os.path.join(data_folder, 'test-labels.gz'), True).reshape(-1)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep = '\n')

    # get hold of the current run
    run = Run.get_context()

    print('Train a logistic regression model with regularization rate of', regularization)
    clf = LogisticRegression(solver='lbfgs', multi_class='auto', C=1.0/regularization, random_state=42)
    clf.fit(X_train, y_train)

    print('Predict the test set')
    y_hat = clf.predict(X_test)

    # calculate accuracy on the prediction
    acc = np.average(y_hat == y_test)
    print('Accuracy is', acc)

    run.log('regularization rate', np.float(regularization))
    run.log('accuracy', np.float(acc))

    os.makedirs('../outputs', exist_ok=True)
    # note file saved in the outputs folder is automatically uploaded into experiment record
    joblib.dump(value=clf, filename=output)

    return clf

if __name__ == '__main__':
    args = parser.parse_args()
    data_folder = args.data_folder
    print('Data folder:', data_folder)
    regularization = args.reg
    print('Regularization rate:', data_folder)
    train(data_folder, regularization, 'outputs/sklearn_mnist_model.pkl')
