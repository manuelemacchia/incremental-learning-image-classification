import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torchvision


def image_grid(images, one_channel=False):
    """Show an image grid.

    Args:
        images (torch.Tensor): batch of images to show
        one_channel (bool): if True, show images in grayscale, otherwise
            show images in RGB (in this case, images tensor should contain
            three channels per image)
    """
    img_grid = torchvision.utils.make_grid(images)

    if one_channel:
        img_grid = img_grid.mean(dim=0)
    img_grid = img_grid / 2 + 0.5  # Unnormalize
    npimg = img_grid.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def train_val_scores(train_loss, train_accuracy, validation_loss, validation_accuracy, save_directory):
    """Plot loss and accuracy for training and validation

    The plot contains two axes:
        axes[0] is the loss plot
        axes[1] is the accuracy plot

    Args:
        train_loss, train_accuracy, validation_loss, validation_accuracy:
            mean and standard deviation with comparable x axis
        save_directory (path): if specified, save the plot as a file in this
            directory
    """

    fig, axes = plt.subplots(1, 2, figsize=[15, 5])

    x = np.arange(10, 101, 10)

    # Use errorbar to plot the standard deviation between different runs
    axes[0].errorbar(x, np.array(train_loss)[:, 0], np.array(train_loss)[:, 1],
                     color='#2E84D5', linewidth=2.5, label='Training')
    axes[0].errorbar(x, np.array(validation_loss)[:, 0], np.array(validation_loss)[:, 1],
                     color='#FF9232', linewidth=2.5, label='Validation')
    axes[0].set_title("Training and validation loss")
    axes[0].set_xlabel("Number of classes")
    axes[0].set_ylabel("Loss")

    axes[1].errorbar(x, np.array(train_accuracy)[:, 0], np.array(train_accuracy)[:, 1],
                     color='#2E84D5', linewidth=2.5, label='Training')
    axes[1].errorbar(x, np.array(validation_accuracy)[:, 0], np.array(validation_accuracy)[:, 1],
                     color='#FF9232', linewidth=2.5, label='Validation')
    axes[1].set_title("Training and validation accuracy")
    axes[1].set_xlabel("Number of classes")
    axes[1].set_ylabel("Accuracy")

    # Layout
    plt.tight_layout()

    axes[0].legend()
    axes[0].grid(True)

    axes[1].legend()
    axes[1].grid(True)

    # Save image if directory is specified
    if save_directory != None:
        fig.savefig(save_directory)

    plt.show()


def test_scores(test_accuracy, save_directory=None):
    """Plot accuracy for test
    
    Args:
        test_accuracy: mean and standard deviation of test accuracy
        save_directory (path): if specified, save the plot as a file in this
            directory
    """

    fig, ax = plt.subplots(1, 1, figsize=[15, 5])

    x = np.arange(10, 101, 10)

    # Use errorbar to plot the standard deviation between different runs
    ax.errorbar(x, np.array(test_accuracy)[:, 0], np.array(test_accuracy)[:, 1],
                  color='#2E84D5', linewidth=2.5)
    ax.set_title("Test accuracy")
    ax.set_xlabel("Number of classes")
    ax.set_ylabel("Accuracy")

    # Layout
    plt.tight_layout()

    ax.legend()
    ax.grid(True)

    # Save image if directory is specified
    if save_directory != None:
        fig.savefig(save_directory)

    plt.show()