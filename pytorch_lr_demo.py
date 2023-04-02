import torch
from torch import nn
import matplotlib.pyplot as plt

# Linear regression demo
weight = 0.7
bias = 0.3

start, end = 0,1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias # linear regression

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


def plot_predictions(train_data, train_label, test_data, test_label, predictions=None):
  plt.figure(figsize=(10, 7))

  plt.scatter(train_data, train_label, c='b', s=4, label="Training Data")
  plt.scatter(test_data, test_label, c='g', s=4, label="Test Data")

  if predictions is not None:
    plt.scatter(test_data, predictions, c='r', s=4, label="Predictions")

  plt.legend(prop={"size": 14})


class LinearRegressionModelV2(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    # Use nn.Linear for creating the parameters
    self.linear_layer = nn.Linear(in_features=1,out_features=1) # taking input of size 1 and output of size 1 because the A and y are 1D array

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear_layer(x)
  

# set the seed
torch.manual_seed(42)
model1 = LinearRegressionModelV2()
model1, model1.state_dict()


# Loss function
loss_fn = nn.L1Loss()

# optimizer
optimizer = torch.optim.SGD(lr=0.01, params=model1.parameters())


# training loop

epochs = 100

for epoch in range(epochs):
  model1.train()

  # forward pass
  y_pred = model1(X_train)

  # calculate loss
  loss = loss_fn(y_pred, y_train)

  # optimizer zero grad
  optimizer.zero_grad()

  # Perform backpropagation
  loss.backward()

  # optimizer step
  optimizer.step()

  # Testing
  model1.eval()

  with torch.inference_mode():
    test_pred = model1(X_test)
    test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:
      print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")


with torch.inference_mode():
  y_pred_test = model1(X_test)
plot_predictions(X_train, y_train, X_test, y_test, y_pred_test)