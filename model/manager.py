import torch
import torch.nn as nn
from torch.backends import cudnn
from tqdm import tqdm

class Manager():
    def __init__(self, device, net, criterion, train_dataloader, val_dataloader, test_dataloader):
        self.device = device
        self.net = net
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def increment_classes(self, n=10):
        """Add n classes in the final fc layer"""
        in_features = self.net.fc.in_features  # size of each input sample
        out_features = self.net.fc.out_features  # size of each output sample
        weight = self.net.fc.weight.data

        self.net.fc = nn.Linear(in_features, out_features+n)
        self.net.fc.weight.data[:out_features] = weight
        self.net.n_classes += n

    def do_batch(self, optimizer, batch, labels):
        """Runs model for one batch."""
        batch = batch.to(self.device)
        labels = labels.to(self.device)

        optimizer.zero_grad()  # Zero-ing the gradients
        outputs = self.net(batch)

        loss = self.criterion(outputs, labels)

        # Get predictions
        _, preds = torch.max(outputs.data, 1)
        running_corrects = torch.sum(
            preds == labels.data).data.item()  # number corrects

        loss.backward()  # backward pass: computes gradients
        optimizer.step()  # update weights based on accumulated gradients

        return (loss, running_corrects)

    def do_epoch(self, optimizer, scheduler, current_epoch):
        """Trains model for one epoch."""

        self.net.train()  # Set network in training mode

        running_train_loss = 0
        running_corrects = 0

        batch_idx = 0
        for images, labels in tqdm(self.train_dataloader, desc='Epoch: %d ' % (current_epoch)):
            loss, corrects = self.do_batch(optimizer, images, labels)
            running_train_loss += loss.item()
            running_corrects += corrects

            batch_idx += 1

        scheduler.step()

        # Average Scores
        train_loss = running_train_loss / batch_idx  # average over all batches
        train_accuracy = running_corrects / \
            len(self.train_dataloader)  # average over all samples

        return (train_loss, train_accuracy)

    def validate(self):
        self.net.train(False)

        running_test_loss = 0
        running_corrects = 0

        batch_idx = 0
        for images, labels in self.val_dataloader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward Pass
            outputs = self.net(images)
            loss = self.criterion(outputs, labels)
            run_val_loss += loss.item()

            # Get predictions
            _, preds = torch.max(outputs.data, 1)

            # Update Corrects
            running_corrects += torch.sum(preds == labels.data).data.item()

            batch_idx += 1

        # Calcuate Scores
        val_loss = running_test_loss / batch_idx
        val_accuracy = running_corrects / float(len(self.val_dataloader))

        return (val_loss, val_accuracy)

    def train(self, optimizer, scheduler, num_epochs):
        self.net.to(self.device)
        cudnn.benchmark  # Calling this optimizes runtime

        train_loss_history = {}
        train_accuracy_history = {}
        val_loss_history = {}
        val_accuracy_history = {}

        for epoch in range(num_epochs):

            train_loss_history[epoch+1], train_accuracy_history[epoch+1] = self.do_epoch(
                optimizer, scheduler, epoch+1)  # Epochs start counting form 1
            # Validate at each epoch
            val_loss_history[epoch +
                             1], val_accuracy_history[epoch+1] = self.validate()

        return (train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history)

    def test(self):

        self.net.train(False)  # Set Network to evaluation mode

        running_corrects = 0
        for images, labels in tqdm(self.test_dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward Pass
            outputs = self.net(images)
            loss = self.criterion(outputs, labels)

            # Get predictions
            _, preds = torch.max(outputs.data, 1)

            # Update Corrects
            running_corrects += torch.sum(preds == labels.data).data.item()

        # Calculate Accuracy
        accuracy = running_corrects / \
            float(len(self.test_dataloader))  # len test dataloader

        print('Test Accuracy: {}'.format(accuracy))

        return accuracy