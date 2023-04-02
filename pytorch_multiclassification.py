import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch import nn

# Set Hyper parameters
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# create multi-class data
X_blob, y_blob = make_blobs(n_samples=1000, n_features=NUM_FEATURES, centers=NUM_CLASSES, cluster_std=1.5, random_state=RANDOM_SEED)

# Turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.float)

# split 
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob, y_blob, test_size=0.2, random_state=RANDOM_SEED)

# plot
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)


class BlobModel(nn.Module):
  def __init__(self, input_features, output_features, hidden_units=8) -> None:
    super().__init__()
    self.linear_layer_stack = nn.Sequential(
        nn.Linear(in_features=input_features, out_features=hidden_units),
        nn.ReLU(),
        nn.Linear(in_features=hidden_units, out_features=hidden_units),
        nn.ReLU(),
        nn.Linear(in_features=hidden_units, out_features=output_features),
    )

  def forward(self, x):
    return self.linear_layer_stack(x)

model = BlobModel(input_features=NUM_FEATURES, output_features=NUM_CLASSES)


# loss function
loss_fn = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


# use accuracy as an evaluation metric instead of just loss
def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct / len(y_pred)) * 100
  return acc


# For multiclass classification to convert our logit outputs to prediction probability we use softmax instead of sigmoid
torch.manual_seed(RANDOM_SEED)

epochs = 100

for epoch in range(epochs):
  # train
  model.train()

  # forward pass
  y_logits = model(X_blob_train)
  y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

  # loss
  loss = loss_fn(y_logits, y_blob_train.type(torch.LongTensor))
  acc = accuracy_fn(y_true=y_blob_train, y_pred=y_pred)

  # optimizer
  optimizer.zero_grad()

  # back propagation
  loss.backward()

  # step
  optimizer.step()

  # Testing
  model.eval()
  with torch.inference_mode():
    # forward pass
    test_logits = model(X_blob_test)
    test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)

    test_loss = loss_fn(test_logits, y_blob_test.type(torch.LongTensor))
    test_acc = accuracy_fn(y_true=y_blob_test, y_pred=test_pred)

  # print
  if epoch % 10 == 0:
    print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")


# Making predictions
model.eval()
with torch.inference_mode():
  y_logits = model(X_blob_test)

# view the first 10 predictions
y_pred_probs = torch.softmax(y_logits, dim=1)
y_pred_probs[:10]
