# This code derived from book titled <Python을 이용한 개인화 추천시스템>
# %%
import pandas as pd
import numpy as np

# csv 파일에서 불러오기
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('/home/jupyter/jupyter-notebooks/machine-learning/ml-100k/u.data', names=r_cols, sep='\t',
                      encoding='latin-1')
ratings = ratings[['user_id', 'movie_id', 'rating']].astype(int)  # timestamp 제거

# train test 분리
from sklearn.utils import shuffle

TRAIN_SIZE = 0.75
ratings = shuffle(ratings)
cutoff = int(TRAIN_SIZE * len(ratings))
ratings_train = ratings.iloc[:cutoff]
ratings_test = ratings.iloc[cutoff:]

# 수정된 부분 1 >>>>>>>>>>
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('/home/jupyter/jupyter-notebooks/machine-learning/ml-100k/u.user', sep='|', names=u_cols,
                    encoding='latin-1')
users = users[['user_id', 'occupation']]

# Convert occupation(string to integer)
occupation = {}


def convert_occ(x):
    if x in occupation:
        return occupation[x]
    else:
        occupation[x] = len(occupation)
        return occupation[x]


users['occupation'] = users['occupation'].apply(convert_occ)

L = len(occupation)
train_occ = pd.merge(ratings_train, users, on='user_id')['occupation']
test_occ = pd.merge(ratings_test, users, on='user_id')['occupation']
# <<<<<<<<< 수정된 부분 1

##### (1)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Add, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam, Adamax

# Variable 초기화
K = 200  # Latent factor 수
mu = ratings_train.rating.mean()  # 전체 평균
M = ratings.user_id.max() + 1  # Number of users
N = ratings.movie_id.max() + 1  # Number of movies


# Defining RMSE measure
def RMSE(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


##### (2)

# Keras model
user = Input(shape=(1,))  # User input
item = Input(shape=(1,))  # Item input
P_embedding = Embedding(M, K, embeddings_regularizer=l2())(user)  # (M, 1, K)
Q_embedding = Embedding(N, K, embeddings_regularizer=l2())(item)  # (N, 1, K)
user_bias = Embedding(M, 1, embeddings_regularizer=l2())(user)  # User bias term (M, 1, )
item_bias = Embedding(N, 1, embeddings_regularizer=l2())(item)  # Item bias term (N, 1, )

# Concatenate layers
from tensorflow.keras.layers import Dense, Concatenate, Activation

P_embedding = Flatten()(P_embedding)  # (K, )
Q_embedding = Flatten()(Q_embedding)  # (K, )
user_bias = Flatten()(user_bias)  # (1, )
item_bias = Flatten()(item_bias)  # (1, )
# R = Concatenate()([P_embedding, Q_embedding, user_bias, item_bias])  # (2K + 2, )

# 수정된 부분 2 >>>>>>>>>>
occ = Input(shape=(1,))
occ_embedding = Embedding(L, 3, embeddings_regularizer=l2())(occ)
occ_layer = Flatten()(occ_embedding)
R = Concatenate()([P_embedding, Q_embedding, user_bias, item_bias, occ_layer])
# <<<<<<<<< 수정된 부분 2

# Neural network
R = Dense(2048)(R)
R = Activation('linear')(R)
R = Dense(256)(R)
R = Activation('linear')(R)
R = Dense(1)(R)

# 수정된 부분 3 >>>>>>>>>>
model = Model(inputs=[user, item, occ], outputs=R)
# <<<<<<<<< 수정된 부분 3
# model = Model(inputs=[user, item], outputs=R)
model.compile(
    loss=RMSE,
    # optimizer=SGD(),
    optimizer=Adam(),
    metrics=[RMSE]
)
model.summary()

# Model fitting
result = model.fit(
    x=[ratings_train.user_id.values, ratings_train.movie_id.values, train_occ.values],
    y=ratings_train.rating.values - mu,
    epochs=30,
    batch_size=512,
    validation_data=(
        [ratings_test.user_id.values, ratings_test.movie_id.values, test_occ.values],
        ratings_test.rating.values - mu
    )
)

# Plot RMSE
import matplotlib.pyplot as plt

plt.plot(result.history['RMSE'], label="Train RMSE")
plt.plot(result.history['val_RMSE'], label="Test RMSE")
plt.xlabel('epoch')
plt.ylabel('RMSE')
plt.legend()
plt.show()

# Prediction
user_ids = ratings_test.user_id.values[0:6]
movie_ids = ratings_test.movie_id.values[0:6]
user_occ = test_occ[0:6]
predictions = model.predict([user_ids, movie_ids, user_occ]) + mu
print("Actuals: \n", ratings_test[0:6])
print()
print("Predictions: \n", predictions)


# 정확도(RMSE)를 계산하는 함수
def RMSE2(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


user_ids = ratings_test.user_id.values
movie_ids = ratings_test.movie_id.values
y_pred = model.predict([user_ids, movie_ids, test_occ]) + mu
y_pred = np.ravel(y_pred, order='C')
y_true = np.array(ratings_test.rating)

RMSE2(y_true, y_pred)
