import matplotlib.pyplot as plt
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

def show_image1(train_loader):
    global count, max_images, fig, axes, images, labels, i, img
    # Initialize a counter
    count = 0
    max_images = 5  # Set the number of images to display
    # Create subplots
    fig, axes = plt.subplots(1, max_images, figsize=(15, 5))  # 1 row, max_images columns

    # Iterate
    for images, labels,stable_height,total_height in train_loader:
        # Iterate through each image
        for i in range(images.size(0)):
            if count >= max_images:
                break


            print(f"Image {count + 1} size: {images[i].size()}")


            img = images[i].numpy().transpose(1, 2, 0)
            # If the image is normalized, denormalize it to restore original pixel values
            img = img * 255
            img = np.clip(img, 0, 255).astype(np.uint8)

            # Print debugging inf
            print(f"Displaying image {count + 1}, label: {labels[i].item()}")

            # Display the image in the subplot
            axes[count].imshow(img)
            axes[count].set_title(f'Label: {labels[i].item()}')
            axes[count].axis('off')  # Hide axis

            count += 1

        if count >= max_images:
            break

    # Display all subplots
    plt.tight_layout()
    plt.show()

def show_image2(train_loader):
    global count, max_images, fig, axes, images, labels, i, img
    # Initialize a counter
    count = 0
    max_images = 5  # Set the number of images to display
    # Create subplots
    fig, axes = plt.subplots(1, max_images, figsize=(15, 5))  # 1 row, max_images columns

    # Iterate
    for images, labels in train_loader:
        # Iterate through each image
        for i in range(images.size(0)):
            if count >= max_images:
                break


            print(f"Image {count + 1} size: {images[i].size()}")


            img = images[i].numpy().transpose(1, 2, 0)  # Convert from [C, H, W] to [H, W, C] for displaying in matplotlib

            # If the image is normalized, denormalize it to restore original pixel values
            img = img * 255
            img = np.clip(img, 0, 255).astype(np.uint8)  # Ensure the values are between 0 and 255, and convert to uint8 type


            print(f"Displaying image {count + 1}, label: {labels[i].item()}")

            # Display the image in the subplot
            axes[count].imshow(img)
            axes[count].set_title(f'Label: {labels[i].item()}')
            axes[count].axis('off')  # Hide axis

            count += 1

        if count >= max_images:
            break

    # Display all subplots
    plt.tight_layout()
    plt.show()



# Print batch information
def print_batch_info(loader, loader_name):
    print(f"\n{loader_name} - First five samples from one batch:")

    # Iterate to get one batch of data
    for batch in loader:
        images, labels, stable_height, total_height = batch  # Unpack images, labels, and other features
        for i in range(min(5, len(images))):  # Print information for the first five samples
            print(f"Sample {i + 1} - Image shape: {images[i].shape}, Label: {labels[i]}")
        break


def plot_loss_curve(log_data):

    epochs_range = range(1, len(log_data["train_loss"]) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, log_data["train_loss"], label="Train Loss", marker='o', linestyle='--')
    plt.plot(epochs_range, log_data["val_loss"], label="Val Loss", marker='s', linestyle='-.')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy_curve(log_data):

    epochs_range = range(1, len(log_data["train_acc"]) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, log_data["train_acc"], label="Train Accuracy", marker='o', linestyle='--')
    plt.plot(epochs_range, log_data["val_acc"], label="Val Accuracy", marker='s', linestyle='-.')
    plt.plot(epochs_range, log_data["train_approx_acc"], label="Train Approx Accuracy", marker='^', linestyle=':')
    plt.plot(epochs_range, log_data["val_approx_acc"], label="Val Approx Accuracy", marker='x', linestyle='-')


    max_train_acc = max(log_data["train_acc"])
    max_val_acc = max(log_data["val_acc"])
    max_train_epoch = log_data["train_acc"].index(max_train_acc) + 1
    max_val_epoch = log_data["val_acc"].index(max_val_acc) + 1


    plt.axvline(max_train_epoch, color='red', linestyle='--', label=f"Max Train Acc @ {max_train_epoch}")
    plt.axvline(max_val_epoch, color='blue', linestyle='--', label=f"Max Val Acc @ {max_val_epoch}")


    plt.text(max_train_epoch, max_train_acc, f'{max_train_acc:.2f}', ha='center', va='bottom', color='red')
    plt.text(max_val_epoch, max_val_acc, f'{max_val_acc:.2f}', ha='center', va='bottom', color='blue')

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()
