'''Try to train a simple neural network using Trust Region/Line Search.'''
import numpy as np
from keras.utils.np_utils import to_categorical
import pylab as plt
import seaborn as sns
from seaborn import xkcd_rgb as xkcd


def sample_data(points_per_class, dimension=2, number_classes=3):
    N = points_per_class # number of points per class
    D = dimension # dimensionality
    K = number_classes # number of classes
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels
    for j in np.arange(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    return X, y


if __name__ =='__main__':

    # Create some testing data.
    X,y = sample_data(5000)
    y_ = to_categorical(y)

    colors = np.argmax(y_,1)
    plt.ion()
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    
