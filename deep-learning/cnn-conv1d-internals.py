# %%
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, GlobalMaxPooling1D, Flatten, Concatenate, merge, concatenate
from keras.layers import Conv1D, Conv2D, Conv3D, Merge, Concatenate

import keras
import numpy as np

# %% Numpy input creates.
np.random.seed(1)
x = np.array([[1, 2, 3, 4, 5], [1, 2, 6, 7, 8]])  # (None, 5)
size_of_the_vocab = 8
y = np.array([np.ones((4, 7)), np.ones((4, 7))])  # (None, 4, 7)

# %% Build a model
model = Sequential()
model.add(Embedding(size_of_the_vocab + 1, output_dim=3,
                    embeddings_initializer='ones',
                    input_shape=(5,)))
model.add(Conv1D(7, kernel_size=2,
                 kernel_initializer='ones',
                 bias_initializer='zeros'))

model.compile(loss=keras.losses.mean_squared_error,
              optimizer='sgd',
              metrics=['accuracy'])

model.summary()

# %%
model.fit(x, y, epochs=1)

# %%
model.predict(x[:1])

# %% Print weights.
names = [weight.name for layer in model.layers for weight in layer.weights]
weights = model.get_weights()

# suppress scientific notation
np.set_printoptions(suppress=True)
for name, weight in zip(names, weights):
    print(name, weight.shape)
    print(weight)

    layer_type = name.split('/')[1]
    if layer_type == 'kernel:0':
        w1 = weight  # (2, 3, 7))
    elif layer_type == 'bias:0':
        b1 = weight  # (7,)

    print()

# %% obtain the embedding and conv output
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.layers[0].output)
embedding_output = intermediate_layer_model.predict(x[:1])
print("embedding_output:", embedding_output)

intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.output)
output = intermediate_layer_model.predict(x[:1])
print("output:", output)


# %% calculate conv1d by hand
eo = embedding_output[0]
a = 0
a += (eo[0, 0] * w1[0, 0, 0])
a += (eo[1, 0] * w1[1, 0, 0])

a += (eo[0, 1] * w1[0, 1, 0])
a += (eo[1, 1] * w1[1, 1, 0])

a += (eo[0, 2] * w1[0, 2, 0])
a += (eo[1, 2] * w1[1, 2, 0])

a += b1[0]

print("expected:", output[0, 0, 0])  # expected
print("actual:", a)
