import numpy as np
import torch
from torchvision import transforms
from torchvision import datasets
from PIL import Image

CLASS_BATCH_SIZE = 10


class Cifar100(torch.utils.data.Dataset):
    def __init__(self, root, train, download, random_state, transform=None):
        self.train = train
        self.transform = transform
        self.is_transform_enabled = True

        self.dataset = datasets.cifar.CIFAR100(
            root=root,
            train=train,
            download=download,
            transform=None)

        self.targets = np.array(self.dataset.targets)
        self.batch_splits = self.class_batches(random_state) # Use class_batches(k:[batch labels]) to build k-th split dataset
    

    def set_classes_batch(self, batch_idx):
        self.batch_idx =  batch_idx

        mask = np.isin(self.targets, self.batch_idx) # Boolean mask returning only indexes where (real) targets match an element of batch_idx 
        idxes = np.where(mask)[0] # batch indices of interest
        

        # fake_idx = index used in __getitem__ to retrieve record of interest
        # real_idx = index used in __getitem__ to return element form self.dataset
        self.batches_mapping = {
            fake_idx: real_idx
            for fake_idx, real_idx in enumerate(idxes)
        }
        self.idxes = np.array(idxes)


    def class_batches(self, random_state):
        batch_splits = dict.fromkeys(np.arange(0, CLASS_BATCH_SIZE)) # {0:None, 1:None, ... , 9:None}

        rs = np.random.RandomState(random_state)
        random_labels =list(range(0,  100)) # [0-99] labels
        rs.shuffle(random_labels) # randomly shuffle the labels

        for i in range(CLASS_BATCH_SIZE):
            # Take 10-sized label batches and define the class splits
            batch_splits[i] = random_labels[i*CLASS_BATCH_SIZE : (i+1)*CLASS_BATCH_SIZE] # {0:[1-st split classes], 1:[...], ... , 99:[...]}
        
        # Label mapping
        self.label_map = {k:v for v,k in enumerate(random_labels)}

        return batch_splits
    

    def set_examplars(self, idxes):
        self.batches_mapping.update({
            fake_idx: real_idx
            for fake_idx, real_idx in zip(range(len(self.batches_mapping), len(idxes)), idxes)
        })

    def train_val_split(self, val_size, random_state):
        len_dataset = len(self.batches_mapping)
        indices = list(range(len_dataset))
        split = int(np.floor(val_size * len_dataset))
        rs = np.random.RandomState(random_state)  # seed the generator
        # shuffle indices to get balanced distribution in training and validation set
        rs.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        return train_indices, val_indices


    def get_true_index(self, fake_idx):
        return self.batches_mapping[fake_idx]

    def __len__(self):
        return len(self.batches_mapping)

    def __getitem__(self, idx):
        real_idx = self.batches_mapping[idx]

        image = self.dataset.data[real_idx]
        label = self.dataset.targets[real_idx]

        image = Image.fromarray(image) # Return a PIL image

        # Applies preprocessing when accessing the image if transformations are currently enabled
        if (self.transform is not None) and (self.is_transform_enabled is True):
            image = self.transform(image)

        mapped_label = self.label_map[label]


        return image, mapped_label

    def enable_transform(self):
        self.is_transform_enabled = True

    def disable_transform(self):
        self.is_transform_enabled = False

    def transform_status(self):
        return self.is_transform_enabled
