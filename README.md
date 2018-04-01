# Jupyter Notebooks
머신러닝, 통계학 기타 여러가지 이론적 기반을 주피터 노트북으로 도식화 하여 정리한다.

scikit-learn, matplotlib, seaborn 등을 이용한 헬퍼 라이브러리를 [kaon-learn](https://github.com/likejazz/kaon-learn)이라는 이름으로 별도로 구현하여 활용하며, 헬퍼에는 decision boundaries를 function으로 처리하여 도식화 하는 등의 기능이 포함되어 있다.

## Machine Learning
- [뉴스 분류](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/news-classification.ipynb)
    - [나이브 베이즈 비교](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/news-classification-nb.ipynb)
    - [다항 분포 나이브 베이즈](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/multinomial-naive-bayes.ipynb)
- [타이타닉 디시젼 트리](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/titanic.ipynb)
- [로지스틱 회귀](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/logistic-regression.ipynb)
- [차원 축소](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/dimensionality-reduction.ipynb)
- [데이타 표현](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/representing-data.ipynb)
- [모델 평가](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/model-evaluation.ipynb)
- [알고리즘 체인과 파이프라인](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/algorithm-chains-and-pipelines.ipynb)

### Support Vector Machine
- [서포트 벡터 머신 분류](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/iris-svm.ipynb)
- [서포트 벡터 머신과 신경망 비교](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/support-vector-machine-explained.ipynb)

## Deep Learning
- [퍼셉트론과 신경망](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/perceptron.ipynb)
- [단어의 벡터 표현](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/vector-representation-of-words.ipynb)
- [게시판 감정 분석](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/sentimental-analysis-word2vec-keras.ipynb)
- [IMDB RNN 감정 분석](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/imdb-sentimental-analysis-rnn.ipynb)
- [선형 대수: 전치 행렬과 미분](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/linear-algebra-transpose-differential.ipynb)
- [역전파 계산](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/backpropagation.ipynb)
- [Keras 중간층 디버깅](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/keras-intermediate-debugging.ipynb)

- [linear-regression-tensorflow.py](deep-learning/linear-regression-tensorflow.py)
tf의 선형 회귀 epoch 단위 표현
- [uci-news.py](deep-learning/uci-news.py)
Kaggle의 uci-news-aggregator 데이터셋 CNN 분류
- [mnist.py](deep-learning/mnist.py)
MNIST fully connected layer tf 구현
- [keras-shape-inspect.py](deep-learning/keras-shape-inspect.py)
Keras 모델의 Merge, Concatenate 검증
- [cnn-conv1d-internals.py](deep-learning/cnn-conv1d-internals.py)
텍스트 임베딩의 Conv1D 검증
- [cnn-conv2d-internals.py](deep-learning/cnn-conv2d-internals.py)
이미지의 `(samples, channels, rows, cols)` 4D tensor Conv2D 검증

## Statistics
- [깁스 샘플링](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/gibbs-sampling.ipynb)
- [정규 분포](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/gaussian-distribution.ipynb)
- [사다리 게임 확률 분석](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/ghost-leg-probabilities.ipynb)

# 참고
- &lt;Introduction to Machine Learning with Python&gt;, Andreas C. Muller
