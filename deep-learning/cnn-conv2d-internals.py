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
x = np.array([np.random.rand(3, 5, 10)])  # (None, 3, 5, 10) channels first
y = np.array([np.ones((7, 4, 9))])  # (None, 7, 4, 9)

# %% Build a model
model = Sequential()
model.add(Conv2D(7, kernel_size=2,
                 data_format='channels_first',
                 kernel_initializer='ones',
                 bias_initializer='zeros',
                 input_shape=(3, 5, 10)))

model.compile(loss=keras.losses.mean_squared_error,
              optimizer='sgd',
              metrics=['accuracy'])

model.summary()

# %%
model.fit(x, y, epochs=1)

# %%
model.predict(x)

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
        w1 = weight  # (2, 2, 3, 7)
    elif layer_type == 'bias:0':
        b1 = weight  # (7,)

    print()

# %% obtain the output
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.output)
output = intermediate_layer_model.predict(x[:1])
print("output:", output)

# %% calculate conv2d by hand
xx = x[0]
a = 0
a += (xx[0, 0, 0] * w1[0, 0, 0, 0])
a += (xx[0, 0, 1] * w1[0, 1, 0, 0])
a += (xx[0, 1, 0] * w1[1, 0, 0, 0])
a += (xx[0, 1, 1] * w1[1, 1, 0, 0])

a += (xx[1, 0, 0] * w1[0, 0, 1, 0])
a += (xx[1, 0, 1] * w1[0, 1, 1, 0])
a += (xx[1, 1, 0] * w1[1, 0, 1, 0])
a += (xx[1, 1, 1] * w1[1, 1, 1, 0])

a += (xx[2, 0, 0] * w1[0, 0, 2, 0])
a += (xx[2, 0, 1] * w1[0, 1, 2, 0])
a += (xx[2, 1, 0] * w1[1, 0, 2, 0])
a += (xx[2, 1, 1] * w1[1, 1, 2, 0])

a += b1[0]

print("expected:", output[0, 0, 0, 0])  # expected
print("actual:", a)
