""" -*- coding: utf-8 -*-
@ Time: 2021/1/18 12:24
@ author: Zhang Chi
@ e-mail: 220200785@mail.seu.edu.cn
@ file: train.py
@ project: U_V_Net_tumour_paper
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:47:08 2019

@author: Sun
"""
import os
import tensorflow as tf
import numpy as np
import h5py
from skimage import io, transform, measure
import scipy.misc
from SISR_tensorflow.forward_srcnn import forward
import time
from SISR_tensorflow.feed import *

# from forward_vdsr import forward
# from forward_drcn import forward
# from forward_drrn import forward
# from forward_LapSRN import forward

dataset_dir = os.path.join(os.getcwd(), 'Dataset')  # h5格式数据集
model_name = 'srcnn'
Learning_rate = 1e-4

data_feeder = DataFeeder(dataset_dir)
one_batch = data_feeder.generate_batch(batch_size=20, num_epochs=40)
print("Read Data Successfully")

X = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name="Input_Low_Res_Image")
Y = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name="High_Res_Image")
iteration_num = tf.placeholder(tf.float32)

pred = forward(X)
l2_loss = tf.reduce_mean(tf.square(pred - Y))
'''
#batch_normalization
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    g_LR2HR_trainner = tf.train.AdadeltaOptimizer(learning_rate=Learning_rate).minimize(l2_loss)
'''

train_op = tf.train.AdadeltaOptimizer(learning_rate=Learning_rate).minimize(l2_loss)

i = 0
g_loss = []
saver = tf.train.Saver()

'''
data = h5py.File(dataset_dir, 'r')
train_set = data['train_set']
label_set = data['label_set']
batch_idxs = len(train_set) // batch_size
steps=0
'''

with tf.Session()as sess:
    tf.global_variables_initializer().run()
    model_dir = os.path.join(os.getcwd(), 'checkpoint')
    model_dir = os.path.join(model_dir, model_name)

    # 以下代码实现断点续训
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(model_dir, ckpt_name))
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    try:
        while True:
            start_time = time.time()
            train_set_epoch, label_set_epoch = sess.run(one_batch)
            _, loss_current, _ = sess.run([train_op, l2_loss, pred],
                                          feed_dict={X: train_set_epoch, Y: label_set_epoch, iteration_num: i})
            g_loss.append(loss_current)
            i += 1
            if i % 2 == 0:  # 每训练2回，打印一次loss值
                print(
                    "Iteration: %d, loss_current: %.6f, Time: %.3f \t\n" % (i, loss_current, time.time() - start_time))

            if i % 10 == 0:  # 每训练10回，保存模型参数并测试
                model_save_path = os.path.join(os.getcwd(), 'checkpoint')
                model_save_path = os.path.join(model_save_path, model_name)
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                saver.save(sess, os.path.join(model_save_path, model_name))

    except tf.errors.OutOfRangeError:
        pass

'''
    for ep in range(epoch):
        for j in range(batch_idxs):

            low_Res = train_set[j*batch_size : (j+1)*batch_size]
            high_Res = label_set[j*batch_size : (j+1)*batch_size]

            low_Res = tf.expand_dims(low_Res[j], axis=0)
            high_Res = tf.expand_dims(high_Res[j], axis=0)

            low_Res = tf.cast(low_Res, dtype=tf.float32)/255
            high_Res = tf.cast(high_Res, dtype=tf.float32)/255

            low_Res=sess.run(low_Res)
            high_Res=sess.run(high_Res)
            prediction, Loss_value, _ = sess.run([pred, l2_loss, g_LR2HR_trainner], feed_dict={X:low_Res, Y:high_Res})
            steps=steps+1
            if steps % 10 == 0:
                print("Epoch: %d, Itereation: %d, Loss value: %f" %((ep+1), steps, Loss_value))

            if steps % 100 == 0:
                model_save_path=os.path.join(os.getcwd(),'checkpoint')
                model_save_path = os.path.join(model_save_path, model_name)
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                #saver.save(sess,os.path.join(model_save_path, 'SRCNN.model'),global_step=steps)
                saver.save(sess,os.path.join(model_save_path, model_name))

print("Successfully")
'''
