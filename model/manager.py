import torch
import torch.nn as nn
from torch.backends import cudnn
from copy import deepcopy

class Manager():
    """Manage training, validation and testing of a neural network.
    
    Args:
        device (string): chosen device to run network operations on
        criterion: loss function
        optimizer: optimization algorithm to change the attributes of the
            neural network, e.g., stochastic gradient descent (SGD)
        scheduler: learning rate scheduling policy, e.g., MultiStepLR
        train_dataloader DataLoader instance of the training set
        val_dataloader: DataLoader instance of the validation set
        test_dataloader: DataLoader instance of the test set
    """

    def __init__(self, device, net, criterion, optimizer, scheduler, train_dataloader, val_dataloader, test_dataloader):
        self.device = device

        self.net = net
        self.best_net = self.net

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    # @todo: do we already initialize to 100 output nodes, as in the iCaRL paper?
    # if so, is increment_classes needed?
    def increment_classes(self, n=10):
        """Add n classes in the final fully connected layer."""

        in_features = self.net.fc.in_features  # size of each input sample
        out_features = self.net.fc.out_features  # size of each output sample
        weight = self.net.fc.weight.data

        self.net.fc = nn.Linear(in_features, out_features+n)
        self.net.fc.weight.data[:out_features] = weight
    
    def output_neurons_count(self):
        """Return the number of output neurons of the current network."""

        return self.net.fc.out_features

    def to_onehot(self, targets): 
      '''
      Args:
      targets : dataloader.dataset.targets of the new task images
      '''
      num_classes = self.net.fc.out_features
      one_hot_targets = torch.eye(num_classes)[targets]

      return one_hot_targets.to(self.device)

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

        self.net.to(self.device)
        cudnn.benchmark  # Calling this optimizes runtime

        self.best_accuracy = 0
        self.best_epoch = 0

        for epoch in range(num_epochs):
            # Run an epoch (start counting form 1)
            train_loss, train_accuracy = self.do_epoch(epoch+1)
        
            # Validate after each epoch 
            val_loss, val_accuracy = self.validate()    

            # Best validation model
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.best_net = deepcopy(self.net)
                self.best_epoch = epoch
                print("Best model updated")

            print("")

        return (train_loss, train_accuracy,
                val_loss, val_accuracy)
    
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

        self.net.train()  # Set network in training mode

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

        return (train_loss, train_accuracy)

    def do_batch(self, batch, labels):
        """Trains model for one batch.
        
        Args:
            batch: batch of images for the model to train.
            labels: labels of the batch of images.
        
        Returns:
            loss: loss function computed on the network outputs of the
                forward pass.
            running_corrects: number of correctly classified images.
        """

        batch = batch.to(self.device)
        labels = labels.to(self.device)

        # Zero-ing the gradients
        self.optimizer.zero_grad() 

        # One hot encoding of new task labels 
        one_hot_labels = self.to_onehot(labels) # Size = [128, 10]

        # New net forward pass
        outputs = self.net(batch)  
        
        loss = self.criterion(outputs, one_hot_labels) # BCE Loss with sigmoids over outputs

        # Get predictions
        _, preds = torch.max(outputs.data, 1)

        # Compute the number of correctly classified images
        running_corrects = \
            torch.sum(preds == labels.data).data.item()

        # Backward pass: computes gradients
        loss.backward()  

        # Update weights based on accumulated gradients
        self.optimizer.step()

        return (loss, running_corrects)

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

        for images, labels in self.val_dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            total += labels.size(0)

            # One hot encoding of new task labels 
            one_hot_labels = self.to_onehot(labels) # Size = [128, 10]
            # New net forward pass
            outputs = self.net(images)  
            loss = self.criterion(outputs, one_hot_labels) # BCE Loss with sigmoids over outputs

            running_val_loss += loss.item()

            # Get predictions
            _, preds = torch.max(outputs.data, 1)

            # Update the number of correctly classified validation samples
            running_corrects += torch.sum(preds == labels.data).data.item()

            batch_idx += 1

        # Calcuate scores
        val_loss = running_val_loss / batch_idx
        val_accuracy = running_corrects / float(total)

        print(f"Validation loss: {val_loss}, Validation accuracy: {val_accuracy}")

        return (val_loss, val_accuracy)

    def test(self):
        """Test the model.
        Returns:
            accuracy (float): accuracy of the model on the test set
        """

        self.best_net.train(False)  # Set Network to evaluation mode

        running_corrects = 0
        total = 0

        all_preds = torch.tensor([]) # to store all predictions
        all_preds = all_preds.type(torch.LongTensor)
        
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

            # Append batch predictions
            all_preds = torch.cat(
                (all_preds.to(self.device), preds.to(self.device)), dim=0
            )

        # Calculate accuracy
        accuracy = running_corrects / float(total)  

        print(f"Test accuracy: {accuracy}")

        return (accuracy, all_preds)
