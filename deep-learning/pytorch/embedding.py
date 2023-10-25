# %%
import torch

torch.manual_seed(42)
torch.set_printoptions(sci_mode=False)

# the embedding layer will randomly initialize word vector and train those vectors along with the model.
embedding = torch.nn.Embedding(1000, 128)
print(embedding(torch.IntTensor([3, 4])))
