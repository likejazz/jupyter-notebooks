# %%
# This code is heavily derived from https://github.com/graykode/nlp-tutorial/blob/master/1-1.NNLM/NNLM-Torch.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

torch.manual_seed(42)

dtype = torch.FloatTensor

sentences = ["i like dog", "i love coffee", "i hate milk"]

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict)  # Total number of Vocabulary

# NNLM Parameters
n_tokens = 2
n_hidden = 2


def make_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]

        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return input_batch, target_batch


class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()

        self.H = nn.Parameter(torch.randn(n_tokens * n_class, n_hidden).type(dtype))
        self.W = nn.Parameter(torch.randn(n_tokens * n_class, n_class).type(dtype))
        self.d = nn.Parameter(torch.randn(n_hidden).type(dtype))
        self.U = nn.Parameter(torch.randn(n_hidden, n_class).type(dtype))
        self.b = nn.Parameter(torch.randn(n_class).type(dtype))

    def forward(self, X):
        # [batch_size, n_step * n_class]
        input = X.view(-1, n_tokens * n_class)
        # [batch_size, n_hidden]
        tanh = nn.functional.tanh(self.d + torch.mm(input, self.H))
        # https://github.com/graykode/nlp-tutorial/pull/4
        # [batch_size, n_class]
        output = self.b + torch.mm(input, self.W) + torch.mm(tanh, self.U)
        return output


model = NNLM()

# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

input_batch, target_batch = make_batch(sentences)
input_batch = Variable(torch.Tensor(input_batch))
target_batch = Variable(torch.LongTensor(target_batch))

# Training
losses = []
for epoch in range(3000):
    optimizer.zero_grad()
    output = model(input_batch)

    # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
    # loss = criterion(output, target_batch)
    loss = criterion(output, torch.eye(n_class)[target_batch])
    losses.append(loss)
    if (epoch + 1) % 100 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

# Predict
predict = model(input_batch).data.max(1, keepdim=True)[1]
print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])

import matplotlib.pyplot as plt

plt.plot(losses)
plt.show()

plt.plot(model(input_batch).data[0].numpy())
plt.plot(model(input_batch).data[1].numpy())
plt.plot(model(input_batch).data[2].numpy())
plt.show()
'''
Epoch: 2500 loss = 0.002825
Epoch: 2600 loss = 0.001818
Epoch: 2700 loss = 0.001145
Epoch: 2800 loss = 0.000704
Epoch: 2900 loss = 0.000423
Epoch: 3000 loss = 0.000247
[['i', 'like'], ['i', 'love'], ['i', 'hate']] -> ['dog', 'coffee', 'milk']
'''