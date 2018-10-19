# %%
import numpy as np
from numpy import argmax

np.random.seed(1337)  # for reproducibility
from keras.models import Model
from keras.layers import Input, Dense, Add, Multiply, Activation
from keras.utils import to_categorical

import matplotlib.pyplot as plt
import pandas as pd

import keras.backend as K

np.set_printoptions(suppress=True)


def get_activations(model, inputs, print_shape_only=False, layer_pos=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_pos is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [model.layers[layer_pos].output]
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations if layer_pos is None else activations[0]


def get_data(n, input_dim, attention_column=1):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    x = np.random.standard_normal(size=(n, input_dim))
    y = np.random.randint(low=0, high=5, size=(n, 1))  # 0 ~ 4
    if attention_column is not None:
        x[:, attention_column] = y[:, 0]

    return x, to_categorical(y)


# --
# %%
input_dim = 32
N = 10000
inputs_1, outputs = get_data(N, input_dim)

inputs = Input(shape=(input_dim,))
# ATTENTION PART STARTS HERE
x = Dense(input_dim, name='attention_vec')(inputs)
x = Activation('softmax')(x)
x = Multiply(name='attention_mul')([inputs, x])
# ATTENTION PART FINISHES HERE
x = Dense(64)(x)
x = Dense(5, activation='softmax')(x)
model = Model(input=[inputs], output=x)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# %%
history = model.fit([inputs_1], outputs, epochs=100, batch_size=128, validation_split=0.1, verbose=2)

# %%
t, to = get_data(1, input_dim, attention_column=1)

# Attention vector corresponds to the second matrix.
# The first one is the Inputs output.
plt.clf()
attention_vector = get_activations(model, t,
                                   print_shape_only=True,
                                   layer_pos=2).flatten()
print('input =', t)
plt.title("input values")
plt.bar(range(input_dim), t[0])
plt.show()

print('attention =', attention_vector)
pd.DataFrame(attention_vector, columns=['attention (%)']).plot(kind='bar',
                                                               title='Attention Mechanism as '
                                                                     'a function of input'
                                                                     ' dimensions. output: ' + str(argmax(to)))
plt.show()

# %%
plt.title("fully connected layer before applying softmax")
plt.bar(range(32), get_activations(model, t, print_shape_only=True, layer_pos=1).flatten())
plt.show()

# %%
plt.title("fully connected layer - Dense(64)")
plt.bar(range(64), get_activations(model, t, print_shape_only=True, layer_pos=4).flatten())
plt.show()
