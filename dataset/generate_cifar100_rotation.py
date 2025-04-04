import numpy as np
import os
import sys
import random
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file

# Set random seeds for reproducibility
random.seed(1)
np.random.seed(1)
num_clients = 100
num_classes = 100
dir_path = "Cifar100_alpha01_100_rotation_15angle/"

# Allocate data to users
def generate_cifar100(dir_path, num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory paths for train/test data and config file
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    # Check if data already exists and is correctly formatted
    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return
        
    # Define transformation for CIFAR-100 data
    transform = transforms.Compose(
        [transforms.ToTensor(), 
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Download and load CIFAR-100 training data
    trainset = torchvision.datasets.CIFAR100(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    # Download and load CIFAR-100 test data
    testset = torchvision.datasets.CIFAR100(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    
    # Create data loaders for training and test data
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    # Extract data and targets from data loaders
    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    # Combine training and test data
    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)
    
    
    # Extract data and labels
    dataset_image = np.concatenate((trainset.data, testset.data), axis=0)
    dataset_label = np.concatenate((np.array(trainset.targets), np.array(testset.targets)), axis=0)

    # Data augmentation: rotation
    rotation_angles = [15, -15]
    augmented_images = []
    augmented_labels = []

    
    # Apply rotation to each image for each specified angle
    for angle in rotation_angles:
        rotation_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation((angle, angle)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        for img, label in zip(dataset_image, dataset_label):
            img = torch.tensor(img)
            rotated_img = rotation_transform(img).numpy()
            augmented_images.append(rotated_img)
            augmented_labels.append(label)
    
    # Concatenate original and augmented data
    dataset_image = np.concatenate((dataset_image, np.array(augmented_images)), axis=0)
    dataset_label = np.concatenate((dataset_label, np.array(augmented_labels)), axis=0)

    # Separate data into clients
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=20)
    # Split data into training and test sets
    train_data, test_data = split_data(X, y)
    # Save the processed data
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)
    

if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_cifar100(dir_path, num_clients, num_classes, niid, balance, partition)
    ## python generate_cifar100.py noniid - dir