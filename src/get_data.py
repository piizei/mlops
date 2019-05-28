import os
from azureml.core import Workspace

from utils.mnist_utils import load_data


def get_data():
    data_folder = '/tmp/azureml_runs/mnist_data/'
    X_train = load_data(os.path.join(data_folder, 'train-images.gz'), False) / 255.0
    y_train = load_data(os.path.join(data_folder, 'train-labels.gz'), True).reshape(-1)

    return {'X': X_train, 'y': y_train}
