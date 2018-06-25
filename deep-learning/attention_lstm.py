# %%
import numpy as np

np.random.seed(1337)  # for reproducibility
from keras.models import Model
from keras.layers import Input, Dense, merge, Flatten, LSTM, \
    Permute, Lambda, RepeatVector, Reshape

from keras_tqdm import TQDMNotebookCallback, TQDMCallback

import matplotlib.pyplot as plt
import pandas as pd

import keras.backend as K

np.set_printoptions(suppress=True)


def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations if layer_name is None else activations[0]


def get_data_recurrent(n, time_steps, input_dim, attention_column=15):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param time_steps: the number of time steps of your series.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    x = np.random.standard_normal(size=(n, time_steps, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column, :] = np.tile(y[:], (1, input_dim))
    return x, y


# --

# %%
INPUT_DIM = 2
TIME_STEPS = 20
# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = False

N = 300000
inputs_1, outputs = get_data_recurrent(N, TIME_STEPS, INPUT_DIM)

if APPLY_ATTENTION_BEFORE_LSTM:
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))  # None, 20, 32
    inputs_dim = inputs.shape[2]

    # -- Start of attention_3d_block
    a = Permute((2, 1))(inputs)
    a = Reshape((int(inputs_dim), TIME_STEPS))(a)
    a = Dense(TIME_STEPS, activation='softmax')(a)

    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(int(inputs_dim))(a)

    a_probs = Permute((2, 1), name='attention_vec')(a)
    attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    # -- End of attention_3d_block

    lstm_units = 32
    # attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)

    model = Model(input=[inputs], output=output)
else:
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))

    lstm_units = 32
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)  # None, 20, 32
    lstm_out_dim = lstm_out.shape[2]

    # -- Start of attention_3d_block
    a = Permute((2, 1))(lstm_out)
    a = Reshape((int(lstm_out_dim), TIME_STEPS))(a)
    a = Dense(TIME_STEPS, activation='softmax')(a)

    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(int(lstm_out_dim))(a)

    a_probs = Permute((2, 1), name='attention_vec')(a)
    attention_mul = merge([lstm_out, a_probs], name='attention_mul', mode='mul')
    # -- End of attention_3d_block

    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)

    model = Model(input=[inputs], output=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# %%
history = model.fit([inputs_1], outputs, epochs=1, batch_size=128, validation_split=0.1, verbose=0,
                    callbacks=[TQDMCallback()])

# %%
attention_vectors = []
for i in range(10):
    t, to = get_data_recurrent(1, TIME_STEPS, INPUT_DIM)
    attention_vector = np.mean(get_activations(model, t,
                                               print_shape_only=True,
                                               layer_name='attention_vec'), axis=2).squeeze()
    print('input =', t)
    print('attention =', attention_vector)
    assert (np.sum(attention_vector) - 1.0) < 1e-5
    attention_vectors.append(attention_vector)

attention_vector_final = np.mean(np.array(attention_vectors), axis=0)

pd.DataFrame(attention_vector_final, columns=['attention (%)']).plot(kind='bar',
                                                                     title='Attention Mechanism as '
                                                                           'a function of input'
                                                                           ' dimensions.')
plt.show()
