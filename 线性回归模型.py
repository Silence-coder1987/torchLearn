import torch
import numpy as np
import torch.nn as nn

x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)
# print(x_train.shape)

y_values = [2 * i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)


# print(y_train.shape)

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  # 用到什么层，就写什么层。

    def forward(self, x):
        out = self.linear(x)  # 自己的前向传播的操作。
        return out


# 定义参数
input_dim, output_dim = 1, 1
model = LinearRegressionModel(input_dim, output_dim)

# 指定GPU训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

print(model)

epochs = 1000
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# 训练模型
for epoch in range(epochs):
    epoch += 1

    # 每一次都转化成tensor
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)

    inputs, labels = inputs.to(device), labels.to(device)

    # 梯度清零每一次迭代
    optimizer.zero_grad()

    # 前向传播
    outputs = model(inputs)

    # 计算损失
    loss = criterion(outputs, labels)

    # 反向传播
    loss.backward()

    # 更新权重参数
    optimizer.step()
    if epoch % 50 == 0:
        print(f"epoch{epoch}，loss{loss.item()}")

# predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()

# 如果想把CUDA tensor格式的数据改成numpy时，
# 需要先将其转换成cpu float-tensor随后再转到numpy格式。
# numpy不能读取CUDA tensor 需要将它转化为 CPU tensor
# 将报错代码self.numpy()改为self.cpu().numpy()即可
predicted = model(torch.from_numpy(x_train).to(device).requires_grad_()).data.cpu().numpy()

# torch.save(model.state_dict(), "model.pkl")
# model.load_state_dict(torch.load("model.pkl"))
