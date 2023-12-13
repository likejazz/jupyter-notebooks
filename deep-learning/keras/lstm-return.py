# %%
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
import numpy as np

# define model
inputs1 = Input(shape=(3, 1))
lstm1 = LSTM(1, unroll=True)(inputs1)
model = Model(inputs=inputs1, outputs=[lstm1])
# define input data
model.compile(loss='mse', optimizer='sgd')

data = np.array([0.1, 0.2, 0.3]).reshape((1, 3, 1))
model.fit(data, [0], epochs=200)

# %%
print(model.predict(data))
