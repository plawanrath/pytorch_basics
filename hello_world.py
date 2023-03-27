#!/usr/bin/python3

'''
Tensors are a specialized data structure that are very similar to arrays and matrices. 
In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.
'''

import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print (f"Sample tensor: {x_data}")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print (f"Sample random tensor from existing tensor: {x_rand}")

# shape is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor.
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: {rand_tensor}")
print(f"Ones Tensor: {ones_tensor}")
print(f"Zeros Tensor: {zeros_tensor}")

# Tensor Attributes
print(f"Tensor shape: {rand_tensor.shape}")
print(f"Tensor Datatypes: {rand_tensor.dtype}")
print(f"Tenros Device: {rand_tensor.device}")
