# %%
# https://weiliu2k.github.io/CITS4012/pytorch/nn_oop.html
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

import torch
import matplotlib.pyplot as plt


# %%
class XOR(torch.nn.Module):
    def __init__(self):
        super(XOR, self).__init__()

        self.fc1 = torch.nn.Linear(2, 2)
        self.fc2 = torch.nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# %%
x_train = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 0]]).float()
y_train = torch.tensor([0, 1, 1, 0]).view(4, 1).float()

x_valid = torch.clone(x_train)
y_valid = torch.clone(y_train)

# %%
torch.manual_seed(42)
torch.set_printoptions(sci_mode=False)

model = XOR()
print(model.state_dict())

# %%
y_pred = model(x_train)
print(y_pred)

# %%
loss_fn = torch.nn.MSELoss(reduction='sum')
loss = loss_fn(y_pred, y_train)
print(loss)

# %%
print(model.fc1.weight.grad)
loss.backward()
print(model.fc1.weight.grad)

# %%
torch.manual_seed(42)
torch.set_printoptions(sci_mode=False)

model = XOR()
print(model.state_dict())

learning_rate = 0.01

fc1_weight1_grads = []
fc1_weight2_grads = []
fc1_bias_grads = []
losses = []

for t in range(20000 + 1):
    y_pred = model(x_train)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    loss = loss_fn(y_pred, y_train)
    losses.append(loss.item())
    if t % 1000 == 0:
        print(t, loss.item())
    if t > 0:
        fc1_weight1_grads.append(model.fc1.weight.grad[0][0].item())
        fc1_weight2_grads.append(model.fc1.weight.grad[0][1].item())
        fc1_bias_grads.append(model.fc1.bias.grad[0].item())

    model.zero_grad()
    loss.backward()

    for f in model.parameters():
        f.data -= learning_rate * f.grad

axes01 = plt.subplot(2, 1, 1)
plt.plot(fc1_weight1_grads)
plt.plot(fc1_weight2_grads)
plt.plot(fc1_bias_grads)
plt.title('weights, biases')
axes02 = plt.subplot(2, 1, 2)
plt.plot(losses)
plt.title('losses')
plt.show()
