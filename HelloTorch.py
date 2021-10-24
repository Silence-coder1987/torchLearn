import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

print("torch.__version__@\n ", torch.__version__)

linear = nn.Linear(5, 3)
print("linear@\n ", linear)

data = torch.randn(2, 5)
print("data@\n ", data)

print("linear(data)@\n ", linear(data))

val = torch.randn(2, 2)
print("val@\n ", val)
print("F.relu(val)@\n ", F.relu(val))

print("F.softmax(val,dim=0)@\n ", F.softmax(val, dim=0))

print("F.log_softmax(val, dim=1)@\n ", F.log_softmax(val, dim=1))
