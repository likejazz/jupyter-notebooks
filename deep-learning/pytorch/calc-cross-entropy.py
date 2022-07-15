# %%
import torch
from torch.nn import functional as F

# %%
torch.manual_seed(2042)

input = torch.randn(5, 3)
# tensor([[-0.0732, -0.2299,  1.5860],
#         [-0.0486,  0.6840,  0.0734],
#         [ 1.2827,  0.8387, -1.2416],
#         [ 1.0546,  0.5926,  0.8586],
#         [-0.2474, -0.2398,  1.7473]])
target = torch.tensor([-1, 0, 1, 1, 2])

# tensor(1.0063)
cost = F.cross_entropy(input, target, ignore_index=-1)

# %%
input_s = input[1:, :]
# tensor([[-0.0486,  0.6840,  0.0734],
#         [ 1.2827,  0.8387, -1.2416],
#         [ 1.0546,  0.5926,  0.8586],
#         [-0.2474, -0.2398,  1.7473]])
input_s_softmax = F.softmax(input_s, dim=1)
# tensor([[0.2375, 0.4941, 0.2683],
#         [0.5808, 0.3726, 0.0465],
#         [0.4078, 0.2569, 0.3352],
#         [0.1069, 0.1077, 0.7855]])
target_s = target[1:]

target_s_onehot = torch.zeros(4, 3)
target_s_onehot = target_s_onehot.scatter(1, target_s.unsqueeze(1), 1)
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])

input_s_log = torch.log(input_s_softmax)
# tensor([[-1.4375, -0.7049, -1.3155],
#         [-0.5433, -0.9872, -3.0675],
#         [-0.8969, -1.3589, -1.0929],
#         [-2.2362, -2.2286, -0.2415]])

cost_s = -(target_s_onehot * input_s_log).sum(dim=1).mean()
# tensor([[-1.4375, -0.0000, -0.0000],
#         [-0.0000, -0.9872, -0.0000],
#         [-0.0000, -1.3589, -0.0000],
#         [-0.0000, -0.0000, -0.2415]])

# tensor(1.0063)
assert cost == cost_s
