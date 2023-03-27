#!/usr/bin/python3

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize, Compose
import matplotlib.pyplot as plt


training_data = datasets.StanfordCars(
    root="data",
    split="train",
    download=True,
    transform=Compose([
        Resize((150, 150)),
        ToTensor(),
    ])
)

test_data = datasets.StanfordCars(
    root="data",
    split="test",
    download=True,
    transform=Compose([
        Resize((150, 150)),
        ToTensor(),
    ])
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Display Image and Label
training_features, training_labels = next(iter(train_dataloader))
print(f"Training feature size: {training_features.size()}")
print(f"Training label sizes: {training_labels.size()}")
img = training_features[0].permute(1, 2, 0)
label = training_labels[0]
plt.imshow(img)
plt.show()
print(f"Label: {label}")