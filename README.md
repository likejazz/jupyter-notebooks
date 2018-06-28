# Jupyter Notebooks and more
머신러닝, 딥러닝, 수학, 통계학등 여러가지 이론적 기반을 주피터 노트북으로 도식화 하여 정리한다. PyCharm 2018의 code cells 지원으로 IPython으로 구현한 코드도 함께 정리한다.

## Machine Learning
- [news-classification.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/news-classification.ipynb) Decision Tree, Random Forest, Naive Bayes
    - [news-classification-nb.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/news-classification-nb.ipynb) Naive Bayes 비교
    - [multinomial-naive-bayes.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/multinomial-naive-bayes.ipynb)
- [titanic.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/titanic.ipynb) 타이타닉 Decision Tree, Random Forest
- [logistic-regression.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/logistic-regression.ipynb)
- [dimensionality-reduction.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/dimensionality-reduction.ipynb)
- [representing-data.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/representing-data.ipynb)
- [model-evaluation.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/model-evaluation.ipynb)
- [algorithm-chains-and-pipelines.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/algorithm-chains-and-pipelines.ipynb)
- [iris-svm.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/iris-svm.ipynb) Support Vector Machine with IRIS dataset
- [support-vector-machine-explained.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/support-vector-machine-explained.ipynb) Comparison beween SVM and Neural Network

## Deep Learning
- [perceptron.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/perceptron.ipynb) Perceptron, Neural Network
- [linear-algebra-transpose-differential.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/linear-algebra-transpose-differential.ipynb)
- [backpropagation.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/backpropagation.ipynb)
- [tensorflow-weight.py](deep-learning/tensorflow-weight.py) TensorFlow의 weight 검증
    - [linear-regression-tensorflow.py](deep-learning/linear-regression-tensorflow.py)
    - [mnist.py](deep-learning/mnist.py) MNIST Dense TensorFlow 구현
- [softmax.py](deep-learning/softmax.py) Softmax, Cross-Entropy Loss
- [word2vec.py](deep-learning/word2vec.py)
- [addition_rnn.py](deep-learning/addition_rnn.py) Learning to Execute의 Addition Task 구현
- [addition_seq2seq.py](deep-learning/addition_seq2seq.py) Seq2Seq w/ Attention
- [attention_dense.py](deep-learning/attention_dense.py)
- [attention_lstm.py](deep-learning/attention_lstm.py)

### Keras
- [vector-representation-of-words.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/vector-representation-of-words.ipynb)
- [sentimental-analysis-word2vec-keras.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/sentimental-analysis-word2vec-keras.ipynb)
- [imdb-sentimental-analysis-rnn.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/imdb-sentimental-analysis-rnn.ipynb)
- [uci-news.py](deep-learning/uci-news.py) Kaggle's  uci-news-aggregator 데이터셋 Classification with CNN
- [keras-intermediate-debugging.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/keras-intermediate-debugging.ipynb) Keras 중간층 디버깅
- [keras-shape-inspect.py](deep-learning/keras-shape-inspect.py) Keras 모델의 Merge, Concatenate 검증
- [cnn-conv1d-internals.py](deep-learning/cnn-conv1d-internals.py) 텍스트 임베딩의 Conv1D 검증
- [cnn-conv2d-internals.py](deep-learning/cnn-conv2d-internals.py) 이미지 4D Tensor Conv2D 검증
- [lstm-keras-inspect.py](deep-learning/lstm-keras-inspect.py) LSTM 계산 검증

## Math, Statistics & Data Science
- [gibbs-sampling.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/gibbs-sampling.ipynb)
- [gaussian-distribution.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/gaussian-distribution.ipynb) Gaussion Distribution(Normal Distribution)
- [ghost-leg-probabilities.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/ghost-leg-probabilities.ipynb) 사다리 게임 확률 분석
- [sympy.py](data-science/sympy.py) Sympy 편미분 계산
- [seaborn.py](data-science/seaborn.py)
- [hard_sigmoid.py](data-science/hard_sigmoid.py)
- [poisson.py](data-science/poisson.py) 푸아송 분포
- [odds-ratio.py](data-science/odds-ratio.py) Odds Ratio 1:3 연결 그래프
- [vanishing-gradient.py](data-science/vanishing-gradient.py)
- [prob-with-permutations.py](data-science/prob-with-permutations.py) 중복 순열의 확률
- [euclidean-v-cosine.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/data-science/euclidean-v-cosine.ipynb) Euclidean Distance vs. Cosine Similarity

# 기타

scikit-learn, matplotlib, seaborn 등을 이용한 헬퍼 라이브러리는 [kaon-learn](https://github.com/likejazz/kaon-learn)이라는 이름으로 별도로 구현하여 활용하며, 헬퍼에는 decision boundaries를 function으로 처리하여 도식화 하는 등의 기능이 포함되어 있다. 해당 라이브러리 및 주피터 노트북의 초안은 Andreas C. Muller의 『Introduction to Machine Learning with Python』 를 많이 참고 했다.
