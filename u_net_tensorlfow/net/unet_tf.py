""" -*- coding: utf-8 -*-
@ Time: 2021/1/18 12:23
@ author: Zhang Chi
@ e-mail: 220200785@mail.seu.edu.cn
@ file: unet_tf.py
@ project: U_V_Net_tumour_paper
"""
import tensorflow as tf
import numpy as np


def forward(X):
    # embedding
    conv0 = tf.layers.conv2d(X, filters=256, kernel_size = 3, strides=1, padding = 'SAME')
    relu0 = tf.nn.relu(conv0)
    conv1 = tf.layers.conv2d(relu0, filters = 256, kernel_size = 3, strides=1, padding = 'SAME')
    relu = tf.nn.relu(conv1)

    # inference
    Infer = [None] * 17
    for i in range(0,16):
        conv = tf.layers.conv2d(relu, filters = 256, kernel_size = 3, strides=1, padding = 'SAME')
        relu = tf.nn.relu(conv)
        Infer[i] = relu

    Infer[16] = X
    # reconstruction
    W = tf.Variable(np.full(fill_value=1.0 / 17, shape=[17], dtype=np.float32),name="LayerWeights")
    W_sum = tf.reduce_sum(W)
    Reconstruct = [None] * 17
    output = [None] * 17
    for i in range(0, 17):
        conv = tf.layers.conv2d(Infer[i], filters = 256, kernel_size = 3, strides=1, padding = 'SAME')
        relu = tf.nn.relu(conv)

        conv = tf.layers.conv2d(relu, filters = 3, kernel_size = 3, strides=1, padding = 'SAME')
        relu = tf.nn.relu(conv)

        Reconstruct[i] = relu
        output[i] = Reconstruct[i]*W[i]/W_sum

    pred = tf.add_n(output)

    return pred





