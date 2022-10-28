---
layout: wiki 
title: Jupyter Notebooks
last-modified: 2022/06/02 14:00:29
---

<!-- TOC -->

- [Deep Learning](#deep-learning)
  - [PyTorch](#pytorch)
  - [Keras](#keras)
  - [NLP](#nlp)
- [Machine Learning](#machine-learning)
  - [Gradient Boosting](#gradient-boosting)
- [Math](#math)
  - [Probability](#probability)
  - [Statistics](#statistics)
- [Data Science](#data-science)
- [Cloud](#cloud)
- [Algorithms](#algorithms)

<!-- /TOC -->

# Deep Learning
- [perceptron.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/deep-learning/perceptron.ipynb) Scikit-Learn's Perceptron, Neural Network, Keras on the IRIS dataset.
- [linear-algebra-transpose-differential.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/data-science/linear-algebra-transpose-differential.ipynb)
- [backpropagation.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/deep-learning/backpropagation.ipynb)
- [tensorflow-weight.py](https://github.com/likejazz/jupyter-notebooks/blob/master/deep-learning/tensorflow-weight.py)
    - [linear-regression-tensorflow.py](https://github.com/likejazz/jupyter-notebooks/blob/master/deep-learning/linear-regression-tensorflow.py)
    - [mnist.py](https://github.com/likejazz/jupyter-notebooks/blob/master/deep-learning/mnist.py) TensorFlow Implementaion on the MNIST dataset.
- [softmax.py](https://github.com/likejazz/jupyter-notebooks/blob/master/deep-learning/softmax.py) Softmax, Cross-Entropy Loss
- [vanishing-gradients.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/data-science/vanishing-gradients.ipynb)

## PyTorch
- [time-series-lstm-pytorch.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/deep-learning/pytorch/time-series-lstm-pytorch.ipynb)
- [pytorch-grad.py](https://github.com/likejazz/jupyter-notebooks/blob/master/deep-learning/pytorch/pytorch-grad.py) Gradients calculation using PyTorch.
- [torch-backprop.py](https://github.com/likejazz/jupyter-notebooks/blob/master/deep-learning/pytorch/torch-backprop.py) Gradients checking using PyTorch.
- [transformer-sentiment-analysis.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/deep-learning/transformer-sentiment-analysis.ipynb) Sentiment Analysis with Transformers(BERT).
- [calc-cross-entropy.py](https://github.com/likejazz/jupyter-notebooks/blob/master/deep-learning/pytorch/calc-cross-entropy.py) Calculate Cross-Entropy from Scratch.

## Keras
- [sin-graph-prediction.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/deep-learning/sin-graph-prediction.ipynb)
- [time-series-prediction-rnn.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/deep-learning/time-series-prediction-rnn.ipynb)
- [keras-intermediate-debugging.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/deep-learning/keras-intermediate-debugging.ipynb) Intermediate Layer Debugging in Keras.
    - [keras-shape-inspect.py](https://github.com/likejazz/jupyter-notebooks/blob/master/deep-learning/keras-shape-inspect.py) Validate Merge, Concatenate methods in Keras.
- [addition_rnn.py](https://github.com/likejazz/jupyter-notebooks/blob/master/deep-learning/addition_rnn.py) Implements Seq2Seq Learning for Performing Addition.
    - [addition_seq2seq.py](https://github.com/likejazz/jupyter-notebooks/blob/master/deep-learning/addition_seq2seq.py) Implements Seq2Seq with Attention for Addition Task.
- [attention_dense.py](https://github.com/likejazz/jupyter-notebooks/blob/master/deep-learning/attention_dense.py) Attention Mechanism
    - [attention_lstm.py](https://github.com/likejazz/jupyter-notebooks/blob/master/deep-learning/attention_lstm.py)
    - [keras-attention](https://github.com/likejazz/jupyter-notebooks/blob/master/deep-learning/keras-attention/) Visualize Attention
- [cnn-conv1d-internals.py](https://github.com/likejazz/jupyter-notebooks/blob/master/deep-learning/cnn-conv1d-internals.py) Validate Conv1D on the Text Embeddings.
    - [cnn-conv2d-internals.py](https://github.com/likejazz/jupyter-notebooks/blob/master/deep-learning/cnn-conv2d-internals.py) Validate Conv2D on the Image dataset.
    - [lstm-keras-inspect.py](https://github.com/likejazz/jupyter-notebooks/blob/master/deep-learning/lstm-keras-inspect.py) Validate LSTM calculation.
    
## NLP
- [nnlm.py](https://github.com/likejazz/jupyter-notebooks/blob/master/deep-learning/nlp/nnlm.py) Implementation of Paper "A Neural Probabilistic Language Model(Bengio et al., 2003)"
- [word2vec.py](https://github.com/likejazz/jupyter-notebooks/blob/master/deep-learning/nlp/word2vec.py)
- [vector-representation-of-words.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/deep-learning/nlp/vector-representation-of-words.ipynb)
- [sentimental-analysis-word2vec-keras.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/deep-learning/nlp/sentimental-analysis-word2vec-keras.ipynb)
- [imdb-sentimental-analysis-rnn.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/deep-learning/nlp/imdb-sentimental-analysis-rnn.ipynb)
- [uci-news.py](https://github.com/likejazz/jupyter-notebooks/blob/master/deep-learning/nlp/uci-news.py) Multi-Class Classification with CNN on the UCI News dataset.
- [lstm-seq2seq.py](https://github.com/likejazz/jupyter-notebooks/blob/master/deep-learning/nlp/lstm-seq2seq.py) Basic Character-Level Seq2Seq Model
- [elmo.py](https://github.com/likejazz/jupyter-notebooks/blob/master/deep-learning/nlp/elmo.py) Sentimental Analysis with ELMo Embeddings.
- [allennlp-tutorial.py](https://github.com/likejazz/jupyter-notebooks/blob/master/deep-learning/nlp/allennlp-tutorial.py) AllenNLP Tutorial
- [cnn-classification.py](https://github.com/likejazz/jupyter-notebooks/blob/master/deep-learning/tensorboard/cnn-classification.py) Text Classification with CNN/TensorBoard
- [bert.py](https://github.com/likejazz/jupyter-notebooks/blob/master/deep-learning/nlp/bert.py) Example codes for BERT article.

# Machine Learning
- [iris-dtreeviz.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/machine-learning/iris-dtreeviz.ipynb) Decision Tree visualization for IRIS dataset.
- [decision-tree-from-scratch.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/machine-learning/decision-tree-from-scratch.ipynb) Decision Tree from a Scratch
- [news-classification.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/machine-learning/news-classification.ipynb) Decision Tree, Random Forest, Naive Bayes on the UCI News dataset.
    - [news-classification-nb.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/machine-learning/news-classification-nb.ipynb) Trains Naive Bayes Classifiers on the UCI News dataset.
    - [multinomial-naive-bayes.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/machine-learning/multinomial-naive-bayes.ipynb) Naive Bayes internals.
- [titanic.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/machine-learning/titanic.ipynb) Decision Tree, Random Forest on the Titanic dataset.
- [linear-regression.py](https://github.com/likejazz/jupyter-notebooks/blob/master/machine-learning/linear-regression.py)
- [logistic-regression.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/machine-learning/logistic-regression.ipynb)
- [dimensionality-reduction.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/machine-learning/dimensionality-reduction.ipynb)
- [representing-data.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/machine-learning/representing-data.ipynb)
- [model-evaluation.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/machine-learning/model-evaluation.ipynb) Decision Regions, Confusion Matrix.
- [algorithm-chains-and-pipelines.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/machine-learning/algorithm-chains-and-pipelines.ipynb)
- [iris-svm-trees.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/machine-learning/iris-svm-trees.ipynb) SVM, Decision Tree, Random Forest on the IRIS dataset.
    - [iris-visualized-by-shap-and-lime.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/machine-learning/iris-visualized-by-shap-and-lime.ipynb) IRIS classification visualized by SHAP and LIME.
    - [keras-lstm-for-imdb-sentiment-classification.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/machine-learning/keras-lstm-for-imdb-sentiment-classification.ipynb) Keras LSTM for IMDb Sentiment Classification visualized by SHAP.
- [support-vector-machine-explained.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/machine-learning/support-vector-machine-explained.ipynb) Comparison between SVM and Neural Network
- [loss-3d.py](https://github.com/likejazz/jupyter-notebooks/blob/master/machine-learning/loss-3d.py) 3D Visualization linear graph with loss value.

## Gradient Boosting
- [gradient-boosting-from-scratch.py](https://github.com/likejazz/jupyter-notebooks/blob/master/machine-learning/gradient-boosting-from-scratch.py) Gradient Boosting from Scratch
- [xgboost.py](https://github.com/likejazz/jupyter-notebooks/blob/master/machine-learning/xgboost-RMSE.py) GridSearchCV with XGBoost

# Math
- [sympy.py](https://github.com/likejazz/jupyter-notebooks/blob/master/data-science/sympy.py) Partial Derivatives using Sympy.
- [hard-sigmoid.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/data-science/hard-sigmoid.ipynb)
- [euclidean-v-cosine.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/data-science/euclidean-v-cosine.ipynb) Euclidean Distance vs. Cosine Similarity
- [calc-entropy.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/data-science/calc-entropy.ipynb) Entropy graph when one probability is high.

## Probability
- [ghost-leg-probabilities.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/data-science/ghost-leg-probabilities.ipynb) Ghost Leg Probabilities.
- [prob-with-permutations.py](https://github.com/likejazz/jupyter-notebooks/blob/master/data-science/prob-with-permutations.py) Probabilities with Duplicate Permutations.

## Statistics
- [gibbs-sampling.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/data-science/gibbs-sampling.ipynb)
- [gaussian-distribution.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/data-science/gaussian-distribution.ipynb) Gaussion Distribution(Normal Distribution)
- [poisson-dist.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/data-science/poisson-dist.ipynb) Poisson Distribution
- [odds-ratio.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/data-science/odds-ratio.ipynb) Odds Ratio 1:3 Relation Graph

# Data Science
- [cudf-tutorial.py](https://github.com/likejazz/jupyter-notebooks/blob/master/machine-learning/rapids/cudf-tutorial.py) RAPIDS cuDF
- [seaborn.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/data-science/seaborn.ipynb)
- [pandas-dask-cudf-comparison-on-gpu.ipynb](https://github.com/likejazz/jupyter-notebooks/blob/master/data-science/pandas-dask-cudf-comparison-on-gpu.ipynb)

# Cloud
- [bigquery-pandas.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/data-science/bigquery-pandas.ipynb)

# Algorithms
- [circular-queue.ipynb](https://nbviewer.jupyter.org/github/likejazz/jupyter-notebooks/blob/master/data-science/circular-queue.ipynb) C++ Implementation.
