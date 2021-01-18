""" -*- coding: utf-8 -*-
@ Time: 2021/1/18 12:23
@ author: Zhang Chi
@ e-mail: 220200785@mail.seu.edu.cn
@ file: feed.py
@ project: U_V_Net_tumour_paper
"""
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import h5py
import os
import glob
import scipy.misc
import skimage
import matplotlib.pyplot as plt
from skimage import io, transform, measure


class DataFeeder:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def file_readline(self):
        data = h5py.File(self.dataset_dir, 'r')
        train_set = data['train_set']
        label_set = data['label_set']
        for i, j in zip(train_set, label_set):
            i = (i / 255).astype(np.float32)
            j = (j / 255).astype(np.float32)
            yield (i, j)

    def generate_batch(self, batch_size, num_epochs=None):
        dataset = tf.data.Dataset.from_generator(self.file_readline, (tf.float32, tf.float32))
        # dataset = dataset.shuffle(20).repeat(num_epochs).batch(batch_size).prefetch(buffer_size=batch_size)
        # shuffle中的参数代表缓冲区大小，越大则越混乱
        # 数据集重复指定次数
        dataset = dataset.shuffle(1000).repeat(num_epochs).batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        out_batch = iterator.get_next()
        return out_batch


class TestDataFeeder:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def file_readline(self):
        image_names = sorted(glob.glob(self.dataset_dir + '/*L.png'))   # 用于测试的图片路径数组
        for k in zip(image_names):
            image_HR = io.imread(k, as_grey=False)  # 用于测试的图片不需要分割，需要降采样处理
            image_LR = scipy.misc.imresize(image_HR, size=0.5, interp='bicubic')
            image_LR = scipy.misc.imresize(image_LR, size=2.0, interp='bicubic')

            image_HR = (image_HR/255).astype(np.float32)
            image_LR = (image_LR / 255).astype(np.float32)
            yield (image_HR, image_LR)

    def generate_batch(self, batch_size, num_epochs=None):
        dataset = tf.data.Dataset.from_generator(self.file_readline,(tf.float32, tf.float32))
        # dataset = dataset.shuffle(20).repeat(num_epochs).batch(batch_size).prefetch(buffer_size=batch_size)
        dataset = dataset.repeat(num_epochs).batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        out_batch = iterator.get_next()
        return out_batch




