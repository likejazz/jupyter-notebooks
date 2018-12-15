# %% Positional Encoding
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

tokens = 10
embeds = 100

m = np.zeros([tokens, embeds])
for i in range(tokens):
    for j in range(embeds):
        if j % 2 == 0:  # even
            m[i, j] = np.sin(i / 10000 ** (j / embeds))
        else:
            m[i, j] = np.cos(i / 10000 ** (int(j / 2) * 2 / embeds))

sns.heatmap(m)
plt.show()

# %% Check nn.Linear results with Matrix Multiplication.
import torch
import torch.nn as nn

a = torch.ones([1, 76])
b = nn.Linear(76, 76)
w = list(b.parameters())

print(b(a) == a @ w[0].transpose(-2, -1) + w[1])

# %% Cosine distance for Various Vectors.
from scipy import spatial
from service.client import BertClient

bc = BertClient(ip='xxx', port=3004, port_out=3005)
sentences = [
    '나 는 너 를 사랑 하다 여',
    '나 는 너 를 사랑 하다 였 다',
    '사랑 누 가 말하다 였 나',
]
(q_length, q_tokens, q_embedding, q_ids) = bc.encode(sentences)

love_1 = q_embedding[0][5]
love_2 = q_embedding[1][5]
love_3 = q_embedding[2][1]

print(spatial.distance.cdist([love_2, love_3], [love_1], metric='cosine'))
print(spatial.distance.cdist([love_2 + love_3], [love_1], metric='cosine'))
print(spatial.distance.cdist([love_2 * 10 + love_3], [love_1], metric='cosine'))
print(spatial.distance.cdist([love_2 + love_3 * 10], [love_1], metric='cosine'))
print(spatial.distance.cdist([(love_2 + love_3) * 10], [love_1], metric='cosine'))
