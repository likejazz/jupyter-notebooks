# %%
"""
This code is heavily derived from word2veclite

ref:
1) https://github.com/cbellei/word2veclite
2) http://www.claudiobellei.com/2018/01/07/backprop-word2vec-python/
"""
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

np.set_printoptions(suppress=True)


def tokenize(corpus):
    """
    Tokenize the corpus of text.
    :param corpus: list containing a string of text (example: ["I like playing football with my friends"])
    :return corpus_tokenized: indexed list of words in the corpus, in the same order as the original corpus
        (the example above would return [[1, 2, 3, 4]])
    :return V: size of vocabulary
    """
    # use for t-SNE visualization
    global tokenizer

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)

    return tokenizer.texts_to_sequences(corpus), len(tokenizer.word_index)


def initialize(V, N):
    """
    Initialize the weights of the neural network.
    :param V: size of the vocabulary
    :param N: size of the hidden layer
    :return: weights W1, W2
    """
    np.random.seed(100)
    W1 = np.random.rand(V, N)
    W2 = np.random.rand(N, V)

    return W1, W2


def corpus2io(corpus_tokenized, V, window_size):
    """Converts corpus text into context and center words
    # Arguments
        corpus_tokenized: corpus text
        window_size: size of context window
    # Returns
        context and center words (arrays)
    """
    for words in corpus_tokenized:
        w = len(words)
        for index, word in enumerate(words):
            contexts = []
            center = []
            s = index - window_size
            e = index + window_size + 1
            contexts.append([words[i] - 1 for i in range(s, e) if 0 <= i < w and i != index])
            center.append(word - 1)

            contexts = contexts[0]  # IMPORTANT: dim reduction

            x = np_utils.to_categorical(contexts, V)
            y = np_utils.to_categorical(center, V)

            yield (x, y.ravel())


def softmax(x):
    """Calculate softmax based probability for given input vector
    # Arguments
        x: numpy array/list
    # Returns
        softmax of input array
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class Word2Vec:
    """
    Python implementation of Word2Vec.

    # Arguments
        method : `str`
            choose method for word2vec (options: 'cbow', 'skipgram')
            [default: 'cbow']
        window_size: `integer`
            size of window [default: 1]
        n_hidden: `integer`
            size of hidden layer [default: 2]
        n_epochs: `integer`
            number of epochs [default: 1]
        learning_rate: `float` [default: 0.1]
        corpus: `str`
            corpus text
    """

    def __init__(self, method='cbow', window_size=1, n_hidden=2, n_epochs=1, corpus='', learning_rate=0.1):
        self.window = window_size
        self.N = n_hidden
        self.n_epochs = n_epochs
        self.corpus = corpus
        self.eta = learning_rate
        if method == 'cbow':
            self.method = self.cbow
        elif method == 'skipgram':
            self.method = self.skipgram
        else:
            raise ValueError("Method not recognized. Aborting.")

    def cbow(self, context, center, W1, W2, loss):
        """
        Implementation of Continuous-Bag-of-Words Word2Vec model
        :param context: all the context words (these represent the inputs)
        :param center: the center word (this represents the label)
        :param W1: weights from the input to the hidden layer
        :param W2: weights from the hidden to the output layer
        :param loss: float that represents the current value of the loss function
        :return: updated weights and loss
        """
        x = np.mean(context, axis=0)
        h = np.dot(W1.T, x)
        u = np.dot(W2.T, h)
        y_pred = softmax(u)

        e = -center + y_pred

        dW2 = np.outer(h, e)
        dW1 = np.outer(x, np.dot(W2, e))

        new_W1 = W1 - self.eta * dW1
        new_W2 = W2 - self.eta * dW2

        loss += -float(u[center == 1]) + np.log(np.sum(np.exp(u)))

        return new_W1, new_W2, loss

    def skipgram(self, context, center, W1, W2, loss):
        """
        Implementation of Skip-Gram Word2Vec model
        :param context: all the context words (these represent the labels)
        :param center: the center word (this represents the input)
        :param W1: weights from the input to the hidden layer
        :param W2: weights from the hidden to the output layer
        :param loss: float that represents the current value of the loss function
        :return: updated weights and loss
        """
        h = np.dot(W1.T, center)
        u = np.dot(W2.T, h)
        y_pred = softmax(u)

        e = np.array([-label + y_pred.T for label in context])

        dW2 = np.outer(h, np.sum(e, axis=0))
        dW1 = np.outer(center, np.dot(W2, np.sum(e, axis=0).T))

        new_W1 = W1 - self.eta * dW1
        new_W2 = W2 - self.eta * dW2

        loss += -2 * np.log(len(context)) \
                - np.sum([u[label == 1] for label in context]) + len(context) * np.log(np.sum(np.exp(u)))

        return new_W1, new_W2, loss

    def predict(self, x, W1, W2):
        """Predict output from input data and weights
        :param x: input data
        :param W1: weights from input to hidden layer
        :param W2: weights from hidden layer to output layer
        :return: output of neural network
        """
        h = np.mean([np.dot(W1.T, xx) for xx in x], axis=0)
        u = np.dot(W2.T, h)

        return softmax(u)

    def run(self):
        """
        Main method of the Word2Vec class.
        :return: the final values of the weights W1, W2 and a history of the value of the loss function vs. epoch
        """
        if len(self.corpus) == 0:
            raise ValueError('You need to specify a corpus of text.')

        corpus_tokenized, V = tokenize(self.corpus)
        W1, W2 = initialize(V, self.N)

        loss_vs_epoch = []
        for e in range(self.n_epochs):
            loss = 0.
            for context, center in corpus2io(corpus_tokenized, V, self.window):
                W1, W2, loss = self.method(context, center, W1, W2, loss)
            loss_vs_epoch.append(loss)

        return W1, W2, loss_vs_epoch


corpus = [
    "I like playing football with my friends",
    "I like football",
    "I like soccer",
    "I like playing soccer",
    "I like playing with my friends",
    "I like my friends",
    "I like friends",
    "football with friends",
    "soccer with friends",
]
w2v = Word2Vec(method="cbow", corpus=corpus,
               window_size=2, n_hidden=5,
               n_epochs=100, learning_rate=0.1)
W1, W2, loss_vs_epoch = w2v.run()

# %%
plt.plot(loss_vs_epoch)
plt.show()

# %%
import matplotlib

matplotlib.rc('font', family='AppleGothic')

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
low_dim_embs = tsne.fit_transform(W1)
for i in range(0, len(tokenizer.word_index)):
    c = 0
    for k in tokenizer.word_index:
        if i == c:
            break
        c += 1
    plt.scatter(low_dim_embs[i, 0], low_dim_embs[i, 1])
    plt.annotate(k,
                 xy=(low_dim_embs[i, 0], low_dim_embs[i, 1]))
# plt.legend(loc='best')
plt.show()
