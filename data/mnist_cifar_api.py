import pickle
import numpy as np
import os
from os.path import join, exists
from mnist import MNIST
# np.random.seed(0)


def unpickle(file):
    """
    unpickles cifar batches from the encoded files. Code from
    https://www.cs.toronto.edu/~kriz/cifar.html
    :param file:
    :return:
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def unpack_cifar(direc, dataset_type):
    """
    the data comes in batches. So this function concatenates the data from the batches
    :param direc: directory where the batches are located
    :return:
    """
    assert exists(direc), "directory does not exist"
    X, y = [], []
    for filename in os.listdir(direc):
        if (dataset_type == 'train' and filename[:5] == 'data_') or \
                (dataset_type == 'valid' and filename[:5] == 'test_'):
            data = unpickle(join(direc, filename))
            X.append(data[b'data'].reshape((10000, 3, 32, 32)))
            y += data[b'labels']
    assert X, "No data was found in '%s'. Are you sure the CIFAR10 data is there?" % direc

    X = np.concatenate(X, 0)
    X = np.transpose(X, (0, 2, 3, 1)).astype(np.float32)
    return X, y


def unpack_mnist(direc, dataset_type):
    """
    Unpack the MNIST data and put them in numpy arrays
    :param direc:
    :return:
    """
    assert exists(direc), "directory does not exist"
    try:
        mndata = MNIST(direc)
        if dataset_type == 'train':
            images, labels = mndata.load_training()
        else:
            images, labels = mndata.load_testing()
    except FileNotFoundError as e:
        print('Make sure that you have downloaded the data and put in %s\n Also make sure that the spelling is correct. \
              the MNIST data comes in t10k-images.idx3-ubyte or t10k-images-idx3-ubyte. We expect the latter' % (direc))
        raise FileNotFoundError(e)
    X_mnist = np.array(images).reshape(len(labels), 28, 28)
    y_mnist = np.array(labels)
    X_mnist = X_mnist.astype(np.float32) / np.max(X_mnist)
    return X_mnist, y_mnist


class Datagen():
    """
    Object to sample the data that we can segment. The sample function combines data
    from MNIST and CIFAR and overlaps them
    """

    def __init__(self, direc_mnist, direc_cifar, dataset_type):
        ## Unpack the data
        X_cifar, _ = unpack_cifar(direc_cifar, dataset_type)
        X_mnist, _ = unpack_mnist(direc_mnist, dataset_type)
        self.data = {}
        self.data['mnist'] = X_mnist
        self.data['cifar'] = X_cifar

    def sample(self, batch_size):
        """
        Samples a batch of data. It randomly inserts the MNIST images into cifar images
        :param batch_size:
        :param norm: indicate wether to normalize the data or not
        :return:
        """
        idx_cifar = np.random.choice(self.data['cifar'].shape[0], batch_size)
        idx_mnist = np.random.choice(self.data['mnist'].shape[0], batch_size)
        im_cifar = self.data['cifar'][idx_cifar]
        im_mnist = self.data['mnist'][idx_mnist][:, ::2, ::2]
        size_mnist = 14

        mnist_mask = np.greater(im_mnist, 0.3, dtype=np.float32)
        im_mnist *= mnist_mask

        width_start = np.random.randint(0, 32 - size_mnist, size=(batch_size))
        height_start = np.random.randint(0, 32 - size_mnist, size=(batch_size))

        color = [255, 0, 255]
        mnist_batch = np.repeat(np.expand_dims(im_mnist, 3), 3, 3) * color

        segm_maps = np.zeros((batch_size, 32, 32))

        for i in range(batch_size):
            im_cifar[i, width_start[i]:width_start[i] + size_mnist, height_start[i]:height_start[i] + size_mnist] += \
            mnist_batch[i]
            segm_maps[i, width_start[i]:width_start[i] + size_mnist, height_start[i]:height_start[i] + size_mnist] += \
            mnist_mask[i]
        im_cifar = np.clip(im_cifar, 0, 255)

        return im_cifar, segm_maps
