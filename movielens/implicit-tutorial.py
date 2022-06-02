# %%
import os
import sys
import time

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz, vstack, hstack, lil_matrix
import implicit
import pickle
from implicit.evaluation import train_test_split, precision_at_k, mean_average_precision_at_k

ratings = pd.read_csv('/home/jupyter/jupyter-notebooks/movielens/ml-1m/ratings.dat', delimiter='::', header=None,
                      names=['user_id', 'movie_id', 'rating', 'timestamp'],
                      usecols=['user_id', 'movie_id', 'rating'], engine='python')

# %%
# using a scalar value (40) to convert ratings from a scale (1-5) to a like/click/view (1)
alpha = 40

sparse_user_item = csr_matrix(([alpha] * len(ratings['movie_id']), (ratings['user_id'], ratings['movie_id'])))
# transposing the item-user matrix to create a user-item matrix
sparse_item_user = sparse_user_item.T.tocsr()
# save the matrices for recalculating user on the fly
save_npz("/home/jupyter/jupyter-notebooks/movielens/sparse_user_item.npz", sparse_user_item)
save_npz("/home/jupyter/jupyter-notebooks/movielens/sparse_item_user.npz", sparse_item_user)


# %%
def map_movies(movie_ids):
    '''takes a list of movie_ids and returns a list of dictionaries with movies information'''
    df = pd.read_csv('/home/jupyter/jupyter-notebooks/movielens/ml-1m/movies.dat', delimiter='::', header=None,
                     names=['movie_id', 'title', 'genre'], engine='python')

    # add years to a new column 'year' and remove them from the movie title
    df['year'] = df['title'].str[-5:-1]
    df['title'] = df['title'].str[:-6]

    # creates an ordered list of dictionaries with the movie information for all movie_ids
    mapped_movies = [df[df['movie_id'] == i].to_dict('records')[0] for i in movie_ids]

    return mapped_movies


def map_users(user_ids):
    '''takes a list of user_ids and returns a list of dictionaries with user information'''
    df = pd.read_csv('/home/jupyter/jupyter-notebooks/movielens/ml-1m/users.dat', delimiter='::', header=None,
                     names=['user_id', 'gender', 'agerange', 'occupation', 'timestamp'], engine='python')
    df = df.drop(['timestamp'], axis=1)

    mapped_users = [df[df['user_id'] == i].to_dict('records')[0] for i in user_ids]

    return mapped_users


# %%

def most_similar_items(item_id, n_similar=10):
    '''computes the most similar items'''
    with open('/home/jupyter/jupyter-notebooks/movielens/model.sav', 'rb') as pickle_in:
        model = pickle.load(pickle_in)

    similar, _ = zip(*model.similar_items(item_id, n_similar)[1:])

    return map_movies(similar)


def most_similar_users(user_id, n_similar=10):
    '''computes the most similar users'''
    sparse_user_item = load_npz("sparse_user_item.npz")

    with open('/home/jupyter/jupyter-notebooks/movielens/model.sav', 'rb') as pickle_in:
        model = pickle.load(pickle_in)

    # similar users gives back [(users, scores)]
    # we want just the users and not the first one, because that is the same as the original user
    similar, _ = zip(*model.similar_users(user_id, n_similar)[1:])

    # orginal users items
    original_user_items = list(sparse_user_item[user_id].indices)

    # # this maps back user_ids to their information, which is useful for visualisation
    similar_users_info = map_users(similar)
    # # now we want to add the items that a similar used has rated
    for user_info in mapped:
        # we create a list of items that correspond to the simillar user ids
        # then compare that in a set operation to the original user items
        # as a last step we add it as a key to the user information dictionary
        user_info['items'] = set(list(sparse_user_item[user_info['user_id']].indices)) & set(original_user_items)

    return similar_users_info


# %%
'''computes p@k and map@k evaluation mettrics and saves model'''
sparse_item_user = load_npz("/home/jupyter/jupyter-notebooks/movielens/sparse_item_user.npz")

train, test = train_test_split(sparse_item_user, train_percentage=0.8)

model = implicit.als.AlternatingLeastSquares(factors=128,
                                             regularization=0.1, iterations=20,
                                             calculate_training_loss=False)
start_time = time.time()
model.fit(train)
print(time.time() - start_time)

with open('/home/jupyter/jupyter-notebooks/movielens/model.sav', 'wb') as pickle_out:
    pickle.dump(model, pickle_out)

# p_at_k = precision_at_k(model, train_user_items=train,
#                         test_user_items=test, K=10)
# m_at_k = mean_average_precision_at_k(model, train, test, K=10)


# %%
user_id = 1
'''recommend N items to user'''
sparse_user_item = load_npz("/home/jupyter/jupyter-notebooks/movielens/sparse_user_item.npz")

with open('/home/jupyter/jupyter-notebooks/movielens/model.sav', 'rb') as pickle_in:
    model = pickle.load(pickle_in)

recommended, _ = zip(*model.recommend(user_id, sparse_user_item))

print(recommended)
import pprint

pprint.pprint(map_movies(recommended))

# %%
most_similar_items(110)


# %%
def recommend_all_users():
    '''recommend N items to all users'''
    sparse_user_item = load_npz("/home/jupyter/jupyter-notebooks/movielens/sparse_user_item.npz")

    with open('/home/jupyter/jupyter-notebooks/movielens/model.sav', 'rb') as pickle_in:
        model = pickle.load(pickle_in)

    # numpy array with N recommendations for each user
    # remove first array, because those are the columns
    all_recommended = model.recommend_all(user_items=sparse_user_item, N=10,
                                          recalculate_user=False, filter_already_liked_items=True)[1:]

    # create a new Pandas Dataframe with user_id, 10 recommendations, for all users
    df = pd.read_csv('/home/jupyter/jupyter-notebooks/movielens/ml-1m/users.dat', delimiter='::', header=None,
                     names=['user_id', 'gender', 'agerange', 'occupation', 'timestamp'], engine='python')
    df = df.drop(['gender', 'agerange', 'occupation', 'timestamp'], axis=1)
    df[['rec1', 'rec2', 'rec3', 'rec4', 'rec5', 'rec6', 'rec7', 'rec8', 'rec9', 'rec10']] = pd.DataFrame(
        all_recommended)
    df.to_pickle("/home/jupyter/jupyter-notebooks/movielens/all_recommended.pkl")

    '''melt dataframe into SQL format for Django model
    melted = df.melt(id_vars=['user_id'], var_name='order', value_name='recommendations',
        value_vars=['rec1', 'rec2', 'rec3', 'rec4', 'rec5', 'rec6', 'rec7', 'rec8', 'rec9', 'rec10'])
    melted['order'] = melted.order.str[3:]
    print(melted.sort_values(by=['user_id', 'order']))
    melted.to_pickle('all_recommended_melted.pkl')
    '''

    return df


def recalculate_user(user_ratings):
    '''adds new user and its liked items to sparse matrix and returns recalculated recommendations'''

    alpha = 40
    m = load_npz('/home/jupyter/jupyter-notebooks/movielens/sparse_user_item.npz')
    n_users, n_movies = m.shape

    ratings = [alpha for i in range(len(user_ratings))]

    m.data = np.hstack((m.data, ratings))
    m.indices = np.hstack((m.indices, user_ratings))
    m.indptr = np.hstack((m.indptr, len(m.data)))
    m._shape = (n_users + 1, n_movies)

    # recommend N items to new user
    with open('/home/jupyter/jupyter-notebooks/movielens/model.sav', 'rb') as pickle_in:
        model = pickle.load(pickle_in)
    recommended, _ = zip(*model.recommend(n_users, m, recalculate_user=True))

    return recommended, map_movies(recommended)
