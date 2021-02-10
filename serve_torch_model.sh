#!/bin/bash
torch-model-archiver --model-name mnist_classifier \
    --version 1.0 \
    --serialized-file ./models/mnist_cnn_epoch_1_lr_0d90.pt \
    --model-file ./models/mnist_cnn_epoch_1_lr_0d90.pt \
