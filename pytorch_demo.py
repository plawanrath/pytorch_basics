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
len(X_train), len(y_train), len(X_test), len(y_test)

def plot_predictions(train_data, train_label, test_data, test_label, predictions=None):
  plt.figure(figsize=(10, 7))

  plt.scatter(train_data, train_label, c='b', s=4, label="Training Data")
  plt.scatter(test_data, test_label, c='g', s=4, label="Test Data")

  if predictions is not None:
    plt.scatter(test_data, predictions, c='r', s=4, label="Predictions")

  plt.legend(prop={"size": 14})


class LinearRegression(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

m0 = LinearRegression()


# Train Model

# Loss function
loss_fn = nn.L1Loss()

# Optimizer
optimizer = torch.optim.SGD(params=m0.parameters(), lr=0.01) # stochastic gradianet descent, parameters and learning rate = how much do you want to adjust parameters


# Training Loop
torch.manual_seed(42)

epochs = 100

for epoch in range(epochs):
  # set model to train mode
  m0.train()

  # forward pass
  y_pred = m0(X_train)

  # Calculate Loss
  loss = loss_fn(y_pred, y_train)
  # print (f"Loss: {loss}")

  # Optimizer zero grad
  optimizer.zero_grad()

  # perform backpropagation on loss w.r.t parameters of the model
  loss.backward()

  # Setp the optimizer (perform gradiant descent)
  optimizer.step()

  m0.eval() # turn off gradient tracking
  
  # Testing loop
  with torch.inference_mode():
    # Do forward
    test_pred = m0(X_test)

    # calculate test loss
    test_loss = loss_fn(test_pred, y_test)

    # print
    if epoch % 10 == 0:
      print(f"Epoch: {epoch}, Test Loss: {test_loss}")


with torch.inference_mode():
  y_preds_new = m0(X_test)
plot_predictions(X_train, y_train, X_test, y_test, y_preds_new)


# Saving a Model in PyTorch

# torch.save - saves model in python pickle format
# torch.load - loads a saved PyTorch object
# load_state_dict - stores the state (parameters) dictionary

# You can save/load the state dict or the entire model
from pathlib import Path

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "my_lr.pt"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# save model state dict
torch.save(obj=m0.state_dict(), f=MODEL_SAVE_PATH)


# Loading a Saved model
loaded_m0 = LinearRegression()

# load
loaded_m0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

loaded_m0.eval()
with torch.inference_mode():
  loaded_model_preds = loaded_m0(X_test)

print(f"{y_preds_new == loaded_model_preds}")