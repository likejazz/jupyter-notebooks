import keras
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Input, Embedding, Concatenate, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D, LSTM, Bidirectional, MaxPool1D, MaxPooling1D

import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from util import make_w2v_embeddings

# --

# Parameters
max_features = 5000
maxlen = 50
gpus = 2
batch_size = 1024 * gpus
embedding_dims = 300
epochs = 10

print('Loading data...')
DATA_FILE = "~/.kaggle/datasets/uciml/news-aggregator-dataset/uci-news-aggregator.csv"
df = pd.read_csv(DATA_FILE)

df['TITLE_n'] = df['TITLE']
df, embeddings = make_w2v_embeddings(df, embedding_dim=embedding_dims, empty_w2v=True)

y = OneHotEncoder().fit_transform(
    LabelEncoder().fit_transform(df['CATEGORY']).reshape(-1, 1)
).toarray()

x_train, x_test, y_train, y_test = train_test_split(df['TITLE_n'], y, test_size=0.1)
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print(len(embeddings), 'embeddings input_dim')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, padding='pre', truncating='post', maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, padding='pre', truncating='post', maxlen=maxlen)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

print('Build model...')
model = Sequential()

inputs = Input(shape=(maxlen,), dtype='int32')
# i = Embedding(max_features,
i = Embedding(len(embeddings),
              embedding_dims,
              input_length=maxlen)(inputs)

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
conv0 = Conv1D(250, 3, activation='relu')(i)
conv0 = Dropout(0.5)(conv0)
conv0 = Conv1D(250, 3, activation='relu')(conv0)
conv0 = Dropout(0.5)(conv0)
conv0 = Conv1D(250, 3, activation='relu')(conv0)
conv0 = Dropout(0.5)(conv0)

conv1 = Conv1D(250, 3, activation='relu')(i)
conv1 = Dropout(0.5)(conv1)
conv1 = Conv1D(250, 3, activation='relu')(conv1)
conv1 = Dropout(0.5)(conv1)
conv1 = Conv1D(250, 3, activation='relu')(conv1)
conv1 = Dropout(0.5)(conv1)

conv2 = Conv1D(250, 3, activation='relu')(i)
conv2 = Dropout(0.5)(conv2)
conv2 = Conv1D(250, 3, activation='relu')(conv2)
conv2 = Dropout(0.5)(conv2)
conv2 = Conv1D(250, 3, activation='relu')(conv2)
conv2 = Dropout(0.5)(conv2)

# LSTM
# x = LSTM(50)(i)
# x = Dropout(0.3)(x)
# out = Dense(1, activation='sigmoid')(x)

x = Concatenate(axis=1)([conv0, conv1, conv2])
# x = Flatten()(x)
x = GlobalMaxPooling1D()(x)
# x = GlobalMaxPooling1D()(conv0)
x = Dropout(0.5)(x)
x = Dense(250)(x)
x = Dropout(0.5)(x)
x = Activation('relu')(x)
x = Dense(4)(x)
out = Activation('softmax')(x)

model = Model(inputs=inputs, outputs=out)

if gpus >= 2:
    model = keras.utils.multi_gpu_model(model, gpus=gpus)
# model.compile(loss='binary_crossentropy',
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
trained = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    verbose=2)

# Plot accuracy
plt.subplot(211)
plt.plot(trained.history['acc'])
plt.plot(trained.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot loss
plt.subplot(212)
plt.plot(trained.history['loss'])
plt.plot(trained.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout(h_pad=1.0)
plt.savefig('history-graph.png')

print(str(trained.history['val_acc'][-1])[:6] +
      "(max: " + str(max(trained.history['val_acc']))[:6] + ")")
print("Done.")
