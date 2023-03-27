#!/usr/bin/python3

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import scipy.io as io


# Downloading Dataset

training_data = datasets.StanfordCars(
    root="data",
    split="train",
    download=True,
    transform=ToTensor(),
)

test_data = datasets.StanfordCars(
    root="data",
    split="test",
    download=True,
    transform=ToTensor(),
)

# Analyzing and Visualizing the dataset
mat_train = io.loadmat('data/stanford_cars/devkit/cars_train_annos.mat')
mat_test = io.loadmat('data/stanford_cars/devkit/cars_test_annos.mat')
meta = io.loadmat('data/stanford_cars/devkit/cars_meta.mat')

labels = [l for l in meta['class_names'][0]]

# Get labels
training_data_labels = list()
for val in mat_train['annotations'][0]:
    label = labels[val[-2][0][0] - 1]
    training_data_labels.append(label)

# Visualization
figure = plt.figure(figsize=(12, 12))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, _ = training_data[sample_idx]
    label = training_data_labels[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.permute(1, 2, 0))
plt.show()

