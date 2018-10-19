# %%
from keras.models import Sequential
from keras.layers import TimeDistributed, Dense

# as the first layer in a model
model = Sequential()
# model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
model.add(Dense(8, input_shape=(10, 16)))
# now model.output_shape == (None, 10, 8)
# model.add(TimeDistributed(Dense(32)))
model.add(Dense(32))

model.compile(loss='mse',
              optimizer='sgd',
              metrics=['accuracy'])

model.summary()
