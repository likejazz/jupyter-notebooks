# Jupyter Notebooks and more
머신러닝, 딥러닝, 수학, 통계학등 여러가지 이론적 기반을 주피터 노트북으로 도식화 하여 정리한다. PyCharm의 code cells 지원으로 IPython으로 구현한 코드도 함께 정리한다.

## Machine Learning
- [news-classification.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/news-classification.ipynb) 뉴스 분류
    - [news-classification-nb.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/news-classification-nb.ipynb) 나이브 베이즈 비교
    - [multinomial-naive-bayes.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/multinomial-naive-bayes.ipynb) 다항분포 나이브 베이즈
- [titanic.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/titanic.ipynb) 타이타닉 디시젼 트리
- [logistic-regression.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/logistic-regression.ipynb) 로지스틱 회귀
- [dimensionality-reduction.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/dimensionality-reduction.ipynb) 차원 축소
- [representing-data.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/representing-data.ipynb) 데이터 표현
- [model-evaluation.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/model-evaluation.ipynb) 모델 평가
- [algorithm-chains-and-pipelines.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/algorithm-chains-and-pipelines.ipynb) 알고리즘 체인과 파이프라인
- [iris-svm.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/iris-svm.ipynb) IRIS 데이터셋의 서포트 벡터 머신 분류
- [support-vector-machine-explained.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/support-vector-machine-explained.ipynb) 서포트 벡터 머신과 뉴럴 네트워크 비교

## Deep Learning
- [perceptron.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/perceptron.ipynb) 퍼셉트론과 신경망
- [linear-algebra-transpose-differential.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/linear-algebra-transpose-differential.ipynb) 선형 대수: 전치 행렬과 미분
- [backpropagation.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/backpropagation.ipynb) 역전파 계산
- [tensorflow-weight.py](deep-learning/tensorflow-weight.py) TensorFlow의 weight 검증
    - [linear-regression-tensorflow.py](deep-learning/linear-regression-tensorflow.py) TensorFlow의 선형 회귀 epoch 단위 표현
    - [mnist.py](deep-learning/mnist.py) MNIST fully connected layer TensorFlow 구현

### Keras
- [vector-representation-of-words.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/vector-representation-of-words.ipynb) 단어의 벡터 표현
- [sentimental-analysis-word2vec-keras.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/sentimental-analysis-word2vec-keras.ipynb) 게시판 감정 분석
- [imdb-sentimental-analysis-rnn.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/imdb-sentimental-analysis-rnn.ipynb) IMDB RNN 감정 분석
- [uci-news.py](deep-learning/uci-news.py) Kaggle의 uci-news-aggregator 데이터셋 CNN 분류
- [keras-intermediate-debugging.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/keras-intermediate-debugging.ipynb) Keras 중간층 디버깅
- [keras-shape-inspect.py](deep-learning/keras-shape-inspect.py) Keras 모델의 Merge, Concatenate 검증
- [cnn-conv1d-internals.py](deep-learning/cnn-conv1d-internals.py) 텍스트 임베딩의 Conv1D 검증
- [cnn-conv2d-internals.py](deep-learning/cnn-conv2d-internals.py) 이미지 4D Tensor Conv2D 검증
- [lstm-keras-inspect.py](deep-learning/lstm-keras-inspect.py) LSTM 계산 검증

## Math, Statistics & Data Science
- [gibbs-sampling.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/gibbs-sampling.ipynb) 깁스 샘플링
- [gaussian-distribution.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/gaussian-distribution.ipynb) 가우시안 분포(정규 분포)
- [ghost-leg-probabilities.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/ghost-leg-probabilities.ipynb) 사다리 게임 확률 분석
- [sympy.py](data-science/sympy.py) Sympy 편미분 계산
- [seaborn.py](data-science/seaborn.py) Matplotlib, Seaborn 사용 예제
- [hard_sigmoid.py](data-science/hard_sigmoid.py) Hard Sigmoid 비교
- [poisson.py](data-science/poisson.py) 푸아송 분포
- [odds-ratio.py](data-science/odds-ratio.py) Odds Ratio 1:3 연결 그래프

# 기타

scikit-learn, matplotlib, seaborn 등을 이용한 헬퍼 라이브러리는 [kaon-learn](https://github.com/likejazz/kaon-learn)이라는 이름으로 별도로 구현하여 활용하며, 헬퍼에는 decision boundaries를 function으로 처리하여 도식화 하는 등의 기능이 포함되어 있다. 해당 라이브러리 및 주피터 노트북의 초안은 Andreas C. Muller의 『Introduction to Machine Learning with Python』 를 많이 참고 했다.
