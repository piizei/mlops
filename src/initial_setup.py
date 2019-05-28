import urllib.request
import os

from utils.aml_util import init_workspace_data

data_folder = os.path.join(os.getcwd(), 'data')
out_folder = os.path.join(os.getcwd(), 'outputs')
os.makedirs(data_folder, exist_ok = True)
os.makedirs(out_folder, exist_ok = True)
print('preparing to download MNIST data')

if os.path.isfile(os.path.join(data_folder, 'train-images.gz')):
    print('MNIST files seem to exist. Skipping downloading. Remove existing files from data/ to redo download')
else:
    urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', filename=os.path.join(data_folder, 'train-images.gz'))
    urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', filename=os.path.join(data_folder, 'train-labels.gz'))
    urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', filename=os.path.join(data_folder, 'test-images.gz'))
    urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', filename=os.path.join(data_folder, 'test-labels.gz'))
    print('Download complete')

init_workspace_data(data_folder)

