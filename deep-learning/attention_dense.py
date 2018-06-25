# %%
import numpy as np

np.random.seed(1337)  # for reproducibility
from keras.models import Model
from keras.layers import Input, Dense, merge

from keras_tqdm import TQDMNotebookCallback, TQDMCallback

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
    y = np.random.randint(low=0, high=2, size=(n, 1))
    if attention_column is not None:
        x[:, attention_column] = y[:, 0]
    return x, y


# --

input_dim = 32
N = 10000
inputs_1, outputs = get_data(N, input_dim)

inputs = Input(shape=(input_dim,))

# ATTENTION PART STARTS HERE
attention_probs = Dense(input_dim, activation='softmax', name='attention_vec')(inputs)
attention_mul = merge([inputs, attention_probs], name='attention_mul', mode='mul')
# ATTENTION PART FINISHES HERE

attention_mul = Dense(64)(attention_mul)
output = Dense(1, activation='sigmoid')(attention_mul)
model = Model(input=[inputs], output=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# %%
history = model.fit([inputs_1], outputs, epochs=30, batch_size=128, validation_split=0.1, verbose=0,
                    callbacks=[TQDMCallback()])

# %%
t, to = get_data(1, input_dim, attention_column=1)

# Attention vector corresponds to the second matrix.
# The first one is the Inputs output.
attention_vector = get_activations(model, t,
                                   print_shape_only=True,
                                   layer_pos=1).flatten()
print('input =', t)
print('attention =', attention_vector)

pd.DataFrame(attention_vector, columns=['attention (%)']).plot(kind='bar',
                                                               title='Attention Mechanism as '
                                                                     'a function of input'
                                                                     ' dimensions.')
plt.show()
