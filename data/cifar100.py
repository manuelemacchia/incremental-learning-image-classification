from torchvision.datasets import VisionDataset

from sklearn.model_selection import train_test_split

import os
import os.path
import sys
import pickle

from PIL import Image
import numpy as np

class CIFAR100(VisionDataset):
    """CIFAR-100 dataset handler.
    
    Args:
        root (string): Root directory of the dataset where directory
            cifar-100-python exists.
        split (string, optional): If 'train', creates dataset from training
            set, otherwise creates from test set.
        transform (callable, optional): A function/transform that takes in a
            PIL image and returns a transformed version.
    """

    base_folder = 'cifar-100-python'

    train_filename = 'train'
    test_filename = 'test'
    meta_filename = 'meta'

    def __init__(self, root, split='train', transform=None):
        super(CIFAR100, self).__init__(root, transform=transform)

        self.split = split

        if split == 'train':
            filename = self.train_filename
        else:
            filename = self.test_filename
        
        # @todo: add integrity checks
        data_path = os.path.join(self.root, self.base_folder, filename)

        with open(data_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            self.data = entry['data']
            self.labels = entry['fine_labels']
        
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # Convert to HWC
        
        self.labels = np.array(self.labels)

        meta_path = os.path.join(self.root, self.base_folder, self.meta_filename)
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f, encoding='latin1')
            self.label_names = meta['fine_label_names']

    def __getitem__(self, index):
        """Access an element through its index.

        Args:
            index (int): Index

        Returns:
            tuple: (image, label) where label is index of the target class.
        """

        image, label = self.data[index], self.labels[index]

        image = Image.fromarray(image) # Return a PIL image

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.data)

    def get_class(self, label):
        """Return the indices of data belonging to the specified label."""
        return np.where(self.labels==label)[0]

    def map_labels(self, label_map):
        """Change dataset labels with a label map.
        
        Args:
            label_map (dict): dictionary mapping all original CIFAR100 labels
                to custom labels.

                e.g., {0: custom_label_0, ..., 99: custom_label_99}
        """

        self.label_map = label_map

        np.vectorize(lambda x: self.label_map[x])(self.labels)

        # @todo: also change the order of self.label_names

    def class_splits(self, steps=10, random_state=None):
        """Split the classes in several sets of equal length and return them."""
        rs = np.random.RandomState(random_state)

        idx = np.arange(len(self.label_names))
        rs.shuffle(idx)

        splits = np.split(idx, steps)

        return splits

    def train_val_split(self, class_splits, val_size=0.5, random_state=None):
        """Perform a train and validation split based on given class splits.

        Args:
            class_splits (list): class split returned by self.class_splits
            val_size (int, float or None): size of the validation set.                

        Returns:
            tuple: (train_indices, val_indices) where each element in the tuple
                is a list of lists.

                train_indices is a list of len(class_splits) lists. Each inner
                list contains the training indices belonging to a class split.
                val_indices, analogously, contains the validation indices
                belonging to a class split.

                e.g., [[0, 1, 2, 3, ..., 99],             <- first class split
                       [100, 101, 12, 103, ..., 199],     <- second class split
                       [200, 201, 22, 203, ..., 299],
                       ...
                       [900, 901, 902, 903, ..., 999]]    <- last class split
        """

        train_indices = []
        val_indices = []

        for i, split in enumerate(class_splits):
            train_indices.append([])
            val_indices.append([])

            for c in split:
                # For each class, split the data into train and test
                idx = self.get_class(c)
                train_idx, val_idx = train_test_split(idx.tolist(), test_size=val_size)

                train_indices[i].extend(train_idx)
                val_indices[i].extend(val_idx)

        return train_indices, val_indices

    def test_split(self, class_splits, random_state=None):
        """Perform a train validation split on the dataset.

        Args:
            class_splits (int): class split returned by self.class_splits
            val_size (int, float or None): size of the validation set.                

        Returns:
            test_indices (list): A list of lists. Analogous to train_test_split.
        """

        test_indices = []

        for i, split in enumerate(class_splits):
            test_indices.append([])

            for c in split:
                idx = self.get_class(c)
                test_indices[i].extend(idx)

        return test_indices