import keras
import numpy as np
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 20000
max_len = 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = np.concatenate((x_train, x_test))
y_train = np.concatenate((y_train, y_test))

x_train = sequence.pad_sequences(x_train, maxlen=max_len)
# x_test = sequence.pad_sequences(x_test, maxlen=max_len)

model = keras.models.Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len, name='embed'))
# model.add(layers.Dense(128, activation='relu',
#                        kernel_regularizer=layers.regularizers.l2(0.01),
#                        bias_regularizer=layers.regularizers.l2(0.01),
#                        ))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(128, activation='relu',
#                        kernel_regularizer=layers.regularizers.l2(0.01),
#                        bias_regularizer=layers.regularizers.l2(0.01),
#                        ))
# model.add(layers.Flatten())
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling1D(5))

model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling1D(5))

model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

callbacks = [
    keras.callbacks.TensorBoard(
        log_dir='logs',
        histogram_freq=1,
    )
]
history = model.fit(x_train, y_train, epochs=10, batch_size=128,
                    validation_split=0.1, callbacks=callbacks)
