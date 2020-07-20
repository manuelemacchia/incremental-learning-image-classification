import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.backends import cudnn

import numpy as np
from copy import deepcopy

class LWF():
    def __init__(self, device, net, old_net, criterion, optimizer, scheduler,
                 train_dataloader, val_dataloader, test_dataloader, num_classes=10):

        self.device = device

        self.net = net
        self.best_net = self.net
        self.old_net = old_net  # None for first ten classes

        # BCE formulation
        # Let x = logits, z = labels. The logistic loss is:
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        self.criterion = BCEWithLogitsLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        # can be incremented ouitside methods in the main, or inside methods
        self.num_classes = num_classes
        self.order = np.arange(100)

        self.sigmoid = nn.Sigmoid()

    def warm_up():
        pass

    def increment_classes(self, n=10):
        """Add n classes in the final fully connected layer."""

        in_features = self.net.fc.in_features  # size of each input sample
        out_features = self.net.fc.out_features  # size of each output sample
        weight = self.net.fc.weight.data

        self.net.fc = nn.Linear(in_features, out_features+n)
        self.net.fc.weight.data[:out_features] = weight

    def to_onehot(self, targets):
        """Convert targets to one-hot encoding (for BCE loss).

        Args:
            targets: dataloader.dataset.targets of the new task images
        """

        one_hot_targets = torch.eye(self.num_classes)[targets]

        return one_hot_targets.to(self.device)

    def do_first_batch(self, batch, labels):
        batch = batch.to(self.device)
        labels = labels.to(self.device)  # new classes labels

        # Zero-ing the gradients
        self.optimizer.zero_grad()

        # One hot encoding of new task labels
        one_hot_labels = self.to_onehot(labels)  # Size = [128, 10]

        # New net forward pass
        outputs = self.net(batch)

        # BCE Loss with sigmoids over outputs
        loss = self.criterion(outputs, one_hot_labels)

        # Get predictions
        _, preds = torch.max(outputs.data, 1)

        # Accuracy over NEW IMAGES, not over all images
        running_corrects = \
            torch.sum(preds == labels.data).data.item()

        # Backward pass: computes gradients
        loss.backward()

        self.optimizer.step()

        return loss, running_corrects

    def do_batch(self, batch, labels):
        batch = batch.to(self.device)
        labels = labels.to(self.device)  # new classes labels

        # Zero-ing the gradients
        self.optimizer.zero_grad()

        # One hot encoding of new task labels
        # Size = [128, n_classes] will be sliced as [:, :self.num_classes-10]
        one_hot_labels = self.to_onehot(labels)
        new_classes = (
            self.order[range(self.num_classes-10, self.num_classes)]).astype(np.int32)
        one_hot_labels = torch.stack(
            [one_hot_labels[:, i] for i in new_classes], axis=1)

        # Old net forward pass
        old_outputs = self.sigmoid(self.old_net(batch))  # Size = [128, 100]
        old_classes = (self.order[range(self.num_classes-10)]).astype(np.int32)
        old_outputs = torch.stack([old_outputs[:, i]
                                   for i in old_classes], axis=1)

        # Combine new and old class targets
        targets = torch.cat((old_outputs, one_hot_labels), 1)

        # New net forward pass
        # Size = [128, 100] comparable with the define targets
        outputs = self.net(batch)
        out_classes = (self.order[range(self.num_classes)]).astype(np.int32)
        outputs = torch.stack([outputs[:, i] for i in out_classes], axis=1)

        # BCE Loss with sigmoids over outputs (over targets must be done manually)
        loss = self.criterion(outputs, targets)

        # Get predictions
        _, preds = torch.max(outputs.data, 1)

        # Accuracy over NEW IMAGES, not over all images
        running_corrects = \
            torch.sum(preds == labels.data).data.item()

        # Backward pass: computes gradients
        loss.backward()

        self.optimizer.step()

        return loss, running_corrects

    def do_epoch(self, current_epoch):
        self.net.train()

        running_train_loss = 0
        running_corrects = 0
        total = 0
        batch_idx = 0

        print(f"Epoch: {current_epoch}, LR: {self.scheduler.get_last_lr()}")

        for images, labels in self.train_dataloader:

            if self.num_classes == 10:
                loss, corrects = self.do_first_batch(images, labels)
            else:
                loss, corrects = self.do_batch(images, labels)

            running_train_loss += loss.item()
            running_corrects += corrects
            total += labels.size(0)
            batch_idx += 1

        self.scheduler.step()

        # Calculate average scores
        train_loss = running_train_loss / batch_idx  # Average over all batches
        train_accuracy = running_corrects / \
            float(total)  # Average over all samples

        print(f"Train loss: {train_loss}, Train accuracy: {train_accuracy}")

        return train_loss, train_accuracy

    def train(self, num_epochs):
        """Train the network for a specified number of epochs, and save
        the best performing model on the validation set.

        Args:
            num_epochs (int): number of epochs for training the network.
        Returns:
            train_loss: loss computed on the last epoch
            train_accuracy: accuracy computed on the last epoch
            val_loss: average loss on the validation set of the last epoch
            val_accuracy: accuracy on the validation set of the last epoch
        """

        # @todo: is the return behaviour intended? (scores of the last epoch)

        self.net = self.net.to(self.device)
        if self.old_net != None:
            self.old_net = self.old_net.to(self.device)
            self.old_net.train(False)

        cudnn.benchmark  # Calling this optimizes runtime

        self.best_loss = float("inf")
        self.best_epoch = 0

        for epoch in range(num_epochs):
                # Run an epoch (start counting form 1)
            train_loss, train_accuracy = self.do_epoch(epoch+1)

            # Validate after each epoch
            val_loss, val_accuracy = self.validate()

            # Best validation model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_net = deepcopy(self.net)
                self.best_epoch = epoch
                print("Best model updated")

            print("")

        return train_loss, train_accuracy, val_loss, val_accuracy

    def validate(self):
        """Validate the model.

        Returns:
            val_loss: average loss function computed on the network outputs
                of the validation set (val_dataloader).
            val_accuracy: accuracy computed on the validation set.
        """

        self.net.train(False)

        running_val_loss = 0
        running_corrects = 0
        total = 0
        batch_idx = 0

        for batch, labels in self.val_dataloader:
            batch = batch.to(self.device)
            labels = labels.to(self.device)
            total += labels.size(0)

            # One hot encoding of new task labels
            # Size = [128, 100] will be sliced as [:, :self.num_classes-10]
            one_hot_labels = self.to_onehot(labels)
            new_classes = (
                self.order[range(self.num_classes-10, self.num_classes)]).astype(np.int32)
            one_hot_labels = torch.stack(
                [one_hot_labels[:, i] for i in new_classes], axis=1)

            if self.num_classes > 10:
                # Old net forward pass
                old_outputs = self.sigmoid(
                    self.old_net(batch))  # Size = [128, 100]
                old_classes = (
                    self.order[range(self.num_classes-10)]).astype(np.int32)
                old_outputs = torch.stack(
                    [old_outputs[:, i] for i in old_classes], axis=1)

                # Combine new and old class targets
                targets = torch.cat((old_outputs, one_hot_labels), 1)

            else:
                targets = one_hot_labels

            # New net forward pass
            # Size = [128, 100] comparable with the define targets
            outputs = self.net(batch)
            out_classes = (
                self.order[range(self.num_classes)]).astype(np.int32)
            outputs = torch.stack([outputs[:, i] for i in out_classes], axis=1)

            # BCE Loss with sigmoids over outputs (over targets must be done manually)
            loss = self.criterion(outputs, targets)

            # Get predictions
            _, preds = torch.max(outputs.data, 1)

            # Update the number of correctly classified validation samples
            running_corrects += torch.sum(preds == labels.data).data.item()
            running_val_loss += loss.item()

            batch_idx += 1

        # Calcuate scores
        val_loss = running_val_loss / batch_idx
        val_accuracy = running_corrects / float(total)

        print(
            f"Validation loss: {val_loss}, Validation accuracy: {val_accuracy}")

        return val_loss, val_accuracy

    def test(self):
        """Test the model.
        Returns:
            accuracy (float): accuracy of the model on the test set
        """

        self.best_net.train(False)  # Set Network to evaluation mode

        running_corrects = 0
        total = 0

        all_preds = torch.tensor([])  # to store all predictions
        all_preds = all_preds.type(torch.LongTensor)
        all_targets = torch.tensor([])
        all_targets = all_targets.type(torch.LongTensor)

        for images, labels in self.test_dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            total += labels.size(0)

            # Forward Pass
            outputs = self.best_net(images)

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

        print(f"Test accuracy: {accuracy}")

        return accuracy, all_targets, all_preds
