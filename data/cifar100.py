from torchvision.datasets import VisionDataset, CIFAR100

from sklearn.model_selection import train_test_split

import os
import os.path
import sys
import pickle

from PIL import Image
import numpy as np
import pandas as pd

CLASS_BATCH_SIZE = 10

class Cifar100(VisionDataset):
    """CIFAR-100 dataset handler.

    Args:
        root (string): Root directory of the dataset where directory
            cifar-100-python exists.
        train (boolean): If True, creates dataset from training set, otherwise
            creates from test set.
        download (boolean): If True, download dataset.
        random_state(integer): Random seed used to define class splits.
        transform (callable, optional): A function/transform that takes in a
            PIL image and returns a transformed version.
    """

    def __init__(self, root, train, download, random_state, transform=None):
        super(Cifar100, self).__init__(root, transform=transform)
        self.dataset = CIFAR100(root=root, train=train,
                                download=download, transform=transform)

        self.labels = np.array(self.dataset.targets)
        self.data = self.dataset.data

        # Use class_batches(k:[batch labels]) to build k-th split dataset
        self.batch_splits = self.class_batches(random_state)

    def make_batch_dataset(self, batch_idx):
        """Args:
                batch_idx = list with ten class labels of the batch
        """

        mask = np.in1d(self.labels, batch_idx)
        # reduce dataset to the batch of interest
        self.batch_data = self.data[mask]
        self.batch_labels = self.labels[mask]

    def class_batches(self, random_state):
        # {0:None, 1:None, ... , 9:None}
        batch_splits = dict.fromkeys(np.arange(0, CLASS_BATCH_SIZE))

        rs = np.random.RandomState(random_state)
        random_labels = list(range(0,  100))  # [0-99] labels
        rs.shuffle(random_labels)  # randomly shuffle the labels

        for i in range(CLASS_BATCH_SIZE):
            # Take 10-sized label batches and define the class splits
            # {0:[1-st split classes], 1:[...], ... , 99:[...]}
            batch_splits[i] = random_labels[i *
                                            CLASS_BATCH_SIZE: (i+1)*CLASS_BATCH_SIZE]

        # Label mapping
        self.label_map = {k: v for v, k in enumerate(random_labels)}

        return batch_splits

    def train_val_split(self, val_size, random_state):
        len_dataset = len(self.batch_labels)
        indices = list(range(len_dataset))
        split = int(np.floor(val_size * len_dataset))
        rs = np.random.RandomState(random_state)  # seed the generator
        # shuffle indices to get balanced distribution in training and validation set
        rs.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        return train_indices, val_indices

    def __getitem__(self, index):
        """Access an element through its index.

        Args:
            index (int): Index

        Returns:
            tuple: (image, label) where label is index of the target class.
        """

        image = self.batch_data[index]
        label = self.batch_labels[index]

        image = Image.fromarray(image)  # Return a PIL image

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        mapped_label = self.label_map[label]

        return image, mapped_label

    def __len__(self):
        """Returns the length of the dataset."""
        
        return len(self.batch_data)
