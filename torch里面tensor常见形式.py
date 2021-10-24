import torch
from torch import tensor

# Scalar
x = tensor(42, )
print(x, x.dim())
print(2 * x, 2 * x.item())

# Vector
v = tensor([1.5, -0.5, 3.0])
print("v @ ", v)
print(v.dim())
print(v.size())

# Matrix
M = tensor([[1., 2.], [3., 4.]])
print(M)

print(M.matmul(M))

print(tensor([1., 0.]).matmul(M))

print(M * M)

print(torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
