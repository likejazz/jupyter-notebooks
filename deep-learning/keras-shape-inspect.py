from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, GlobalMaxPooling1D, Flatten
from keras.datasets import imdb

import numpy as np

max_features = 20
maxlen = 5  # cut texts after this number of words (among top max_features most common words)
batch_size = 128
epochs = 2

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

i = 10
x_train = x_train[:i]
y_train = y_train[:i]
x_test = x_test[:i]
y_test = y_test[:i]

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
inputs = Input(shape=(maxlen,), dtype='int32')
x = Embedding(max_features, output_dim=3, trainable=False)(inputs)
# x = LSTM(1, dropout=0.2)(x)
x = Flatten()(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

# print weights
names = [weight.name for layer in model.layers for weight in layer.weights]
weights = model.get_weights()

# suppress scientific notation
np.set_printoptions(suppress=True)
for name, weight in zip(names, weights):
    print(name, weight.shape)
    print(weight)

# obtain the output of an intermediate layer
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('embedding_1').output)
intermediate_output = intermediate_layer_model.predict(x_test[:1])

print("Input:", x_test[:1])
print("Intermediate output:", intermediate_output)
print('Predict value:', model.predict(x_test[:1]))
