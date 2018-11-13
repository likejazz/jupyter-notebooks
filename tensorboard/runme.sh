#!/usr/bin/env bash

rm -rf logs
python cnn-classification.py
tensorboard --logdir=./logs --port 3003
