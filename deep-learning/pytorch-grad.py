# %%
import torch

torch.manual_seed(42)
W_h = torch.randn(30, 20, requires_grad=True)
W_x = torch.randn(30, 10, requires_grad=True)

x = torch.rand(1, 10)
prev_h = torch.rand(1, 20)

h2h = torch.mm(W_h, prev_h.t())
i2h = torch.mm(W_x, x.t())
next_h = h2h + i2h
next_h_tanh = next_h.tanh()

loss = next_h_tanh.sum()
loss.backward()

print('W_h[0, 0]', W_h[0, 0])
print('W_h.grad[0, 0]', W_h.grad[0, 0])

# W_h[0, 0] tensor(1.9269, grad_fn=<SelectBackward>)
# W_h.grad[0, 0] tensor(0.0212)
