import torch
import torch.nn as nn

torch.manual_seed(13)
torch.set_printoptions(sci_mode=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


net = Net()
net.fc1.register_forward_hook(get_activation('fc1'))
net.fc2.register_forward_hook(get_activation('fc2'))

input = torch.randn(2)
output = net(input)

target = torch.tensor([1, 1, 1], dtype=torch.float32)
criterion = nn.MSELoss()
loss = criterion(output, target)

net.zero_grad()
loss.backward()

# w1
"""
-0.0104
( \
    -2 * (1 - -0.0766) * -0.3493 + \
    -2 * (1 - -0.2679) * 0.4791 + \
    -2 * (1 -  0.0754) * -0.2662 \
) / 3 * -1.0575
"""