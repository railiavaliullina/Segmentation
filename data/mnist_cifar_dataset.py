import os
import pickle
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import torch

from sys import stdout
from data.mnist_cifar_api import Datagen


class MNISTCIFARSegmentation(data.Dataset):
    """`MS Coco Captions <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, config, dataset_type, transform=None, img_transform=None, target_transform=None):
        self.config = config
        if dataset_type == 'train':
            cifar_dataset = self.config.datasets.train.root
            mnist_dataset = self.config.datasets.train.annFile
        else:
            cifar_dataset = self.config.datasets.valid.root
            mnist_dataset = self.config.datasets.valid.annFile
        weights_path = self.config.weights_path
        self.datagen = Datagen(mnist_dataset, cifar_dataset, dataset_type)
        self.ids = np.arange(len(self.datagen.data['cifar']))
        self.transform = transform
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.normalization = None
        # self.denorm = UnNormalize(self.config.mean, self.config.std)
        self.denorm = transforms.Normalize(
            mean=[-m / s for m, s in zip(self.config.mean, self.config.std)],
            std=[1 / s for s in self.config.std]
        )
        if dataset_type == 'train':
            if os.path.exists(weights_path):
                with open(weights_path, 'rb') as f:
                    self.W = pickle.load(f)
            else:
                self.W = np.zeros(self.config.num_classes)
                print('Computing class weights matrix...')
                for i in range(len(self.ids)):
                    stdout.write('\r%d/%d' % (i + 1, len(self.ids)))
                    stdout.flush()
                    _, target = self.datagen.sample(1)
                    values, counts = np.unique(target, return_counts=True)
                    self.W[values.astype(np.int32)] += counts
                with open(weights_path, 'wb') as f:
                    pickle.dump(self.W, f)
                print()
            self.W = np.array([1.0, 1.0])
            # self.W /= np.sum(self.W)
            # self.W = 1 - self.W
        self.fixed_img, self.fixed_target = self.datagen.sample(1)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """

        imgs, targets = self.datagen.sample(1)
        # imgs, targets = self.fixed_img, self.fixed_target
        img = imgs[0].astype(np.uint8)
        target = targets[0].astype(np.int64)

        img_padded = np.pad(img, (self.config.pad, self.config.pad, (0, 0)), mode='reflect')

        self.img_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(torch.tensor(self.config.mean),
                                                         torch.tensor(self.config.std))])

        if self.transform is not None:
            img, target = self.transform([img, target])

        if self.img_transform is not None:
            img_padded = self.img_transform(img_padded)
            # img = self.img_transform(img)

        if self.normalization is not None:
            img = self.normalization(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img_padded, img), target.copy()

    def __len__(self):
        return len(self.ids)
