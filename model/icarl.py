import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.backends import cudnn

from math import floor
from copy import deepcopy
import random

sigmoid = nn.Sigmoid() # Sigmoid function

class Exemplars(torch.utils.data.Dataset):
    def __init__(self, exemplars, transform=None):
        # exemplars = [
        #     [ex0_class0, ex1_class0, ex2_class0, ...],
        #     [ex0_class1, ex1_class1, ex2_class1, ...],
        #     ...
        #     [ex0_classN, ex1_classN, ex2_classN, ...]
        # ]

        self.dataset = []
        self.targets = []

        for y, exemplar_y in enumerate(exemplars):
            self.dataset.extend(exemplar_y)
            self.targets.extend([y] * len(exemplar_y))

        self.transform = transform
    
    def __getitem__(self, index):
        image = self.dataset[index]
        target = self.targets[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.targets)

class iCaRL:
    """Implement iCaRL, a strategy for simultaneously learning classifiers and a
    feature representation in the class-incremental setting.
    """

    def __init__(self, device, net, lr, momentum, weight_decay, milestones, gamma, num_epochs, batch_size, train_transform, test_transform):
        self.device = device
        self.net = net

        # Set hyper-parameters
        self.LR = lr
        self.MOMENTUM = momentum
        self.WEIGHT_DECAY = weight_decay
        self.MILESTONES = milestones
        self.GAMMA = gamma
        self.NUM_EPOCHS = num_epochs
        self.BATCH_SIZE = batch_size
        
        # Set transformations
        self.train_transform = train_transform
        self.test_transform = test_transform

        # List of exemplar sets. Each set contains memory_size/num_classes exemplars
        # with num_classes the number of classes seen until now by the network.
        self.exemplars = []

        # Initialize the copy of the old network, used to compute outputs of the
        # previous network for the distillation loss, to None. This is useful to
        # correctly apply the first function when training the network for the
        # first time.
        self.old_net = None

        # Maximum number of exemplars
        self.memory_size = 2000
    
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # If True, test on the best model found (e.g., minimize loss). If False,
        # test on the last model build (of the last epoch).
        self.VALIDATE = False

    def classify(self, batch, train_dataset=None):
        """Mean-of-exemplars classifier used to classify images into the set of
        classes observed so far.

        Args:
            batch (torch.tensor): batch to classify
        Returns:
            label (int): class label assigned to the image
        """

        batch_features = self.extract_features(batch) # (batch size, 64)
        for i in range(batch_features.size(0)):
            batch_features[i] = batch_features[i]/batch_features[i].norm() # Normalize sample feature representation
        batch_features = batch_features.to(self.device)

        if self.cached_means is None:
            print("Computing mean of exemplars... ", end="")

            self.cached_means = []

            # Number of known classes
            num_classes = len(self.exemplars)

            # Compute the means of classes with all the data available,
            # including training data which contains samples belonging to
            # the latest 10 classes. This will remove noise from the mean
            # estimate, improving the results.
            if train_dataset is not None:
                train_features_list = [[] for _ in range(10)]

                for train_sample, label in train_dataset:
                    features = self.extract_features(train_sample, batch=False, transform=self.test_transform)
                    features = features/features.norm()
                    train_features_list[label % 10].append(features)

            # Compute means of exemplars for all known classes
            for y in range(num_classes):
                if (train_dataset is not None) and (y in range(num_classes-10, num_classes)):
                    features_list = train_features_list[y % 10]
                else:
                    features_list = []

                for exemplar in self.exemplars[y]:
                    features = self.extract_features(exemplar, batch=False, transform=self.test_transform)
                    features = features/features.norm() # Normalize the feature representation of the exemplar
                    features_list.append(features)
                
                features_list = torch.stack(features_list)
                class_means = features_list.mean(dim=0)
                class_means = class_means/class_means.norm() # Normalize the class means

                self.cached_means.append(class_means)
            
            self.cached_means = torch.stack(self.cached_means).to(self.device)
            print("done")

        preds = []
        for i in range(batch_features.size(0)):
            f_arg = torch.norm(batch_features[i] - self.cached_means, dim=1)
            preds.append(torch.argmin(f_arg))
        
        return torch.stack(preds)
    
    def extract_features(self, sample, batch=True, transform=None):
        """Extract features from single sample or from batch.
        
        Args:
            sample (PIL image or torch.tensor): sample(s) from which to
                extract features
            batch (bool): if True, sample is a torch.tensor containing a batch
                of images with dimensions (batch_size, 3, 32, 32)
            transform: transformations to apply to the PIL image before
                processing
        Returns:
            features: torch.tensor, 1-D of dimension 64 for single samples or
                2-D of dimension (batch_size, 64) for batch
        """

        assert not (batch is False and transform is None), "if a PIL image is passed to extract_features, a transform must be defined"

        self.net.train(False)
        if self.best_net is not None: self.best_net.train(False)
        if self.old_net is not None: self.old_net.train(False)

        if batch is False: # Treat sample as single PIL image
            sample = transform(sample)
            sample = sample.unsqueeze(0) # https://stackoverflow.com/a/59566009/6486336

        sample = sample.to(self.device)

        if self.VALIDATE:
            features = self.best_net.features(sample)
        else:
            features = self.net.features(sample)

        if batch is False:
            features = features[0]

        return features

    def incremental_train(self, split, train_dataset, val_dataset):
        """Adjust internal knowledge based on the additional information
        available in the new observations.

        Args:
            split (int): current split number, counting from zero
            train_dataset: dataset for training the model
            val_dataset: dataset for validating the model
        Returns:
            train_logs: tuple of four metrics (train_loss, train_accuracy,
            val_loss, val_accuracy) obtained during network training
        """

        if split is not 0:
            # Increment the number of output nodes for the new network by 10
            self.increment_classes(10)

        # Improve network parameters upon receiving new classes. Effectively
        # train a new network starting from the current network parameters.
        train_logs = self.update_representation(train_dataset, val_dataset)

        # Compute the number of exemplars per class
        num_classes = self.output_neurons_count()
        m = floor(self.memory_size / num_classes)

        print(f"Target number of exemplars per class: {m}")
        print(f"Target total number of exemplars: {m*num_classes}")

        # Reduce pre-existing exemplar sets in order to fit new exemplars
        for y in range(len(self.exemplars)):
            self.exemplars[y] = self.reduce_exemplar_set(self.exemplars[y], m)

        # Construct exemplar set for new classes
        new_exemplars = self.construct_exemplar_set_rand(train_dataset, m)
        self.exemplars.extend(new_exemplars)

        return train_logs

    def update_representation(self, train_dataset, val_dataset):
        """Update the parameters of the network.

        Args:
            train_dataset: dataset for training the model
            val_dataset: dataset for validating the model
        Returns:
            train_logs: tuple of four metrics (train_loss, train_accuracy,
            val_loss, val_accuracy) obtained during network training
        """

        # Combine the new training data with existing exemplars.
        print(f"Length of exemplars set: {sum([len(self.exemplars[y]) for y in range(len(self.exemplars))])}")
        exemplars_dataset = Exemplars(self.exemplars, self.train_transform)
        train_dataset_with_exemplars = ConcatDataset([exemplars_dataset, train_dataset])

        # Train the network on combined dataset
        train_logs = self.train(train_dataset_with_exemplars, val_dataset) # @todo: include exemplars in validation set?

        # Keep a copy of the current network in order to compute its outputs for
        # the distillation loss while the new network is being trained.
        self.old_net = deepcopy(self.net)

        return train_logs

    def construct_exemplar_set_rand(self, dataset, m):
        """Randomly sample m elements from a dataset without replacement.

        Args:
            dataset: dataset containing a split (samples from 10 classes) from
                which to take exemplars
            m (int): target number of exemplars per class
        Returns:
            exemplars: list of samples extracted from the dataset
        """

        dataset.dataset.disable_transform()

        samples = [[] for _ in range(10)]
        for image, label in dataset:
            label = label % 10 # Map labels to 0-9 range
            samples[label].append(image)

        dataset.dataset.enable_transform()

        exemplars = [[] for _ in range(10)]

        for y in range(10):
            print(f"Randomly extracting exemplars from class {y} of current split... ", end="")

            # Randomly choose m samples from samples[y] without replacement
            exemplars[y] = random.sample(samples[y], m)

            print(f"Extracted {len(exemplars[y])} exemplars.")

        return exemplars

    def reduce_exemplar_set(self, exemplar_set, m):
        """Procedure for removing exemplars from a given set.

        Args:
            exemplar_set (set): set of exemplars belonging to a certain class
            m (int): target number of exemplars
        Returns:
            exemplar_set: reduced exemplar set
        """

        return exemplar_set[:m]

    def train(self, train_dataset, val_dataset):
        """Train the network for a specified number of epochs, and save
        the best performing model on the validation set.
        
        Args:
            train_dataset: dataset for training the model
            val_dataset: dataset for validating the model
        Returns: train_logs: tuple of four metrics (train_loss, train_accuracy,
            val_loss, val_accuracy) obtained during network training. If
            validation is enabled, return scores of the best epoch, otherwise
            return scores of the last epoch.
        """

        # Define the optimization algorithm
        parameters_to_optimize = self.net.parameters()
        self.optimizer = optim.SGD(parameters_to_optimize, 
                                   lr=self.LR,
                                   momentum=self.MOMENTUM,
                                   weight_decay=self.WEIGHT_DECAY)
        
        # Define the learning rate decaying policy
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones=self.MILESTONES,
                                                        gamma=self.GAMMA)

        # Create DataLoaders for training and validation
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

        # Send networks to chosen device
        self.net = self.net.to(self.device)
        if self.old_net is not None: self.old_net = self.old_net.to(self.device)

        cudnn.benchmark  # Calling this optimizes runtime

        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0
        self.best_train_loss = float('inf')
        self.best_train_accuracy = 0
        
        self.best_net = None
        self.best_epoch = -1

        for epoch in range(self.NUM_EPOCHS):
            # Run an epoch (start counting form 1)
            train_loss, train_accuracy = self.do_epoch(epoch+1)
        
            # Validate after each epoch 
            val_loss, val_accuracy = self.validate()    

            # Validation criterion: best net is the one that minimizes the loss
            # on the validation set.
            if self.VALIDATE and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_accuracy = val_accuracy
                self.best_train_loss = train_loss
                self.best_train_accuracy = train_accuracy

                self.best_net = deepcopy(self.net)
                self.best_epoch = epoch
                print("Best model updated")

        if self.VALIDATE:
            val_loss = self.best_val_loss
            val_accuracy = self.best_val_accuracy
            train_loss = self.best_train_loss
            train_accuracy = self.best_train_accuracy

            print(f"Best model found at epoch {self.best_epoch+1}")

        return train_loss, train_accuracy, val_loss, val_accuracy
    
    def do_epoch(self, current_epoch):
        """Trains model for one epoch.
        
        Args:
            current_epoch (int): current epoch number (begins from 1)
        Returns:
            train_loss: average training loss over all batches of the
                current epoch.
            train_accuracy: training accuracy of the current epoch over
                all samples.
        """

        # Set the current network in training mode
        self.net.train()
        if self.old_net is not None: self.old_net.train(False)
        if self.best_net is not None: self.best_net.train(False)

        running_train_loss = 0
        running_corrects = 0
        total = 0
        batch_idx = 0

        print(f"Epoch: {current_epoch}, LR: {self.scheduler.get_last_lr()}")

        for images, labels in self.train_dataloader:
            loss, corrects = self.do_batch(images, labels)

            running_train_loss += loss.item()
            running_corrects += corrects
            total += labels.size(0)
            batch_idx += 1

        self.scheduler.step()

        # Calculate average scores
        train_loss = running_train_loss / batch_idx # Average over all batches
        train_accuracy = running_corrects / float(total) # Average over all samples

        print(f"Train loss: {train_loss}, Train accuracy: {train_accuracy}")

        return train_loss, train_accuracy

    def do_batch(self, batch, labels):
        """Train network for a batch. Loss is applied here.

        Args:
            batch: batch of data used for training the network
            labels: targets of the batch
        Returns:
            loss: output of the criterion applied
            running_corrects: number of correctly classified elements
        """
        batch = batch.to(self.device)
        labels = labels.to(self.device)

        # Zero-ing the gradients
        self.optimizer.zero_grad()
        
        # One-hot encoding of labels of the new training data (new classes)
        # Size: batch size (rows) by number of classes seen until now (columns)
        #
        # e.g., suppose we have four images in a batch, and each incremental
        #   step adds three new classes. At the second step, the one-hot
        #   encoding may return the following tensor:
        #
        #       tensor([[0., 0., 0., 1., 0., 0.],   # image 0 (label 3)
        #               [0., 0., 0., 0., 1., 0.],   # image 1 (label 4)
        #               [0., 0., 0., 0., 0., 1.],   # image 2 (label 5)
        #               [0., 0., 0., 0., 1., 0.]])  # image 3 (label 4)
        #
        #   The first three elements of each vector will always be 0, as the
        #   new training batch does not contain images belonging to classes
        #   already seen in previous steps.
        #
        #   The last three elements of each vector will contain the actual
        #   information about the class of each image (one-hot encoding of the
        #   label). Therefore, we slice the tensor and remove the columns 
        #   related to old classes (all zeros).
        num_classes = self.output_neurons_count() # Number of classes seen until now, including new classes
        one_hot_labels = self.to_onehot(labels)[:, num_classes-10:num_classes]

        if self.old_net is None:
            # Network is training for the first time, so we only apply the
            # classification loss.
            targets = one_hot_labels

        else:
            # Old net forward pass. We compute the outputs of the old network
            # and apply a sigmoid function. These are used in the distillation
            # loss. We discard the output of the new neurons, as they are not
            # considered in the distillation loss.
            old_net_outputs = sigmoid(self.old_net(batch))[:, :num_classes-10]

            # Concatenate the outputs of the old network and the one-hot encoded
            # labels along dimension 1 (columns).
            # 
            # Each row refers to an image in the training set, and contains:
            # - the output of the old network for that image, used by the
            #   distillation loss
            # - the one-hot label of the image, used by the classification loss
            targets = torch.cat((old_net_outputs, one_hot_labels), dim=1)

        # Forward pass
        outputs = self.net(batch)
        loss = self.criterion(outputs, targets)

        # Get predictions
        _, preds = torch.max(outputs.data, 1)

        # Accuracy over NEW IMAGES, not over all images
        running_corrects = torch.sum(preds == labels.data).data.item() 

        # Backward pass: computes gradients
        loss.backward()

        self.optimizer.step()

        return loss, running_corrects

    def validate(self):
        """Validate the model.
        
        Returns:
            val_loss: average loss function computed on the network outputs
                of the validation set (val_dataloader).
            val_accuracy: accuracy computed on the validation set.
        """

        self.net.train(False)
        if self.old_net is not None: self.old_net.train(False)
        if self.best_net is not None: self.best_net.train(False)

        running_val_loss = 0
        running_corrects = 0
        total = 0
        batch_idx = 0

        for images, labels in self.val_dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            total += labels.size(0)

            # One hot encoding of new task labels 
            one_hot_labels = self.to_onehot(labels)

            # New net forward pass
            outputs = self.net(images)  
            loss = self.criterion(outputs, one_hot_labels) # BCE Loss with sigmoids over outputs

            running_val_loss += loss.item()

            # Get predictions
            _, preds = torch.max(outputs.data, 1)

            # Update the number of correctly classified validation samples
            running_corrects += torch.sum(preds == labels.data).data.item()

            batch_idx += 1

        # Calculate scores
        val_loss = running_val_loss / batch_idx
        val_accuracy = running_corrects / float(total)

        print(f"Validation loss: {val_loss}, Validation accuracy: {val_accuracy}")

        return val_loss, val_accuracy

    def test(self, test_dataset, train_dataset=None):
        """Test the model.

        Args:
            test_dataset: dataset on which to test the network
            train_dataset: training set used to train the last split, if
                available
        Returns:
            accuracy (float): accuracy of the model on the test set
        """

        self.net.train(False)
        if self.best_net is not None: self.best_net.train(False)  # Set Network to evaluation mode
        if self.old_net is not None: self.old_net.train(False)

        self.test_dataloader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4)

        running_corrects = 0
        total = 0

        # To store all predictions
        all_preds = torch.tensor([])
        all_preds = all_preds.type(torch.LongTensor)
        all_targets = torch.tensor([])
        all_targets = all_targets.type(torch.LongTensor)

        # Clear mean of exemplars cache
        self.cached_means = None
        
        # Disable transformations for train_dataset, if available, as we will
        # need original PIL images from which to extract features.
        if train_dataset is not None: train_dataset.dataset.disable_transform()

        for images, labels in self.test_dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            total += labels.size(0)
            
            with torch.no_grad():
                preds = self.classify(images, train_dataset)

            running_corrects += torch.sum(preds == labels.data).data.item()

            all_targets = torch.cat(
                (all_targets.to(self.device), labels.to(self.device)), dim=0
            )

            all_preds = torch.cat(
                (all_preds.to(self.device), preds.to(self.device)), dim=0
            )

        if train_dataset is not None: train_dataset.dataset.enable_transform()

        # Calculate accuracy
        accuracy = running_corrects / float(total)  

        print(f"Test accuracy (iCaRL): {accuracy} ", end="")

        if train_dataset is None:
            print("(only exemplars)")
        else:
            print("(exemplars and training data)")

        return accuracy, all_targets, all_preds

    def test_without_classifier(self, test_dataset):
        """Test the model without classifier, using the outputs of the
        network instead.

        Args:
            test_dataset: dataset on which to test the network
        Returns:
            accuracy (float): accuracy of the model on the test set
        """

        self.net.train(False)
        if self.best_net is not None: self.best_net.train(False) # Set Network to evaluation mode
        if self.old_net is not None: self.old_net.train(False)

        self.test_dataloader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4)

        running_corrects = 0
        total = 0

        all_preds = torch.tensor([]) # to store all predictions
        all_preds = all_preds.type(torch.LongTensor)
        all_targets = torch.tensor([])
        all_targets = all_targets.type(torch.LongTensor)
        
        for images, labels in self.test_dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            total += labels.size(0)

            # Forward Pass
            with torch.no_grad():
                if self.VALIDATE:
                    outputs = self.best_net(images)
                else:
                    outputs = self.net(images)

            # Get predictions
            _, preds = torch.max(outputs.data, 1)

            # Update Corrects
            running_corrects += torch.sum(preds == labels.data).data.item()

            all_targets = torch.cat(
                (all_targets.to(self.device), labels.to(self.device)), dim=0
            )

            # Append batch predictions
            all_preds = torch.cat(
                (all_preds.to(self.device), preds.to(self.device)), dim=0
            )

        # Calculate accuracy
        accuracy = running_corrects / float(total)  

        print(f"Test accuracy (hybrid1): {accuracy}")

        return accuracy, all_targets, all_preds
    
    def increment_classes(self, n=10):
        """Add n classes in the final fully connected layer."""

        in_features = self.net.fc.in_features  # size of each input sample
        out_features = self.net.fc.out_features  # size of each output sample
        weight = self.net.fc.weight.data
        bias = self.net.fc.bias.data

        self.net.fc = nn.Linear(in_features, out_features+n)
        self.net.fc.weight.data[:out_features] = weight
        self.net.fc.bias.data[:out_features] = bias
    
    def output_neurons_count(self):
        """Return the number of output neurons of the current network."""

        return self.net.fc.out_features
    
    def feature_neurons_count(self):
        """Return the number of neurons of the last layer of the feature extractor."""

        return self.net.fc.in_features
    
    def to_onehot(self, targets):
        """Convert targets to one-hot encoding (for BCE loss).

        Args:
            targets: dataloader.dataset.targets of the new task images
        """
        num_classes = self.net.fc.out_features
        one_hot_targets = torch.eye(num_classes)[targets]

        return one_hot_targets.to(self.device)

    def network_params(self):
        weight = self.net.fc.weight.data
        bias = self.net.fc.bias.data

        return weight, bias
