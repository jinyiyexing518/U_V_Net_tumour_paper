""" -*- coding: utf-8 -*-
@ Time: 2021/1/15 18:15
@ author: Zhang Chi
@ e-mail: 220200785@mail.seu.edu.cn
@ file: uv_scale_attention.py
@ project: U_V_Net_tumour_paper
"""
import sys
import codecs
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
import keras
from keras.models import *
from keras.layers import Input, Conv3D, Conv2D, Multiply, Deconvolution3D, Dropout, Concatenate, GlobalAveragePooling3D, Lambda, Reshape
from keras.optimizers import *
from keras import layers
from keras import backend as K

from keras.callbacks import ModelCheckpoint
from uv_net_scale_attention.net.fit_generator import get_path_list, get_train_batch
import matplotlib.pyplot as plt
import tensorflow as tf

net_name = "uvnet_scale_attention"
train_order = 1
last_order = 0

train_batch_size = 1
epoch = 5
img_size = 400

data_train_path = "../../v_net_25D/net/vnet_3_1_input/train"
data_label_path = "../../v_net_25D/net/vnet_3_1_input/label"
train_path_list, label_path_list, count = get_path_list(data_train_path, data_label_path)


# 写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('dice_coef'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('dice_coef'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train dice')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('dice-loss')
        plt.legend(loc="best")
        plt.show()


# class WeightedBinaryCrossEntropy(object):
#
#     def __init__(self, pos_ratio=0.7):
#         neg_ratio = 1. - pos_ratio
#         self.pos_ratio = tf.constant(pos_ratio, tf.float32)
#         self.weights = tf.constant(neg_ratio / pos_ratio, tf.float32)
#         self.__name__ = "weighted_binary_crossentropy({0})".format(pos_ratio)
#
#     def __call__(self, y_true, y_pred):
#         return self.weighted_binary_crossentropy(y_true, y_pred)
#
#     def weighted_binary_crossentropy(self, y_true, y_pred):
#         # Transform to logits
#         epsilon = tf.convert_to_tensor(K.common._EPSILON, y_pred.dtype.base_dtype)
#         y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
#         y_pred = tf.log(y_pred / (1 - y_pred))
#
#         cost = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, self.weights)
#         return K.mean(cost * self.pos_ratio, axis=-1)


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def mycrossentropy(y_true, y_pred, e=0.1):
    nb_classes = 10
    loss1 = K.categorical_crossentropy(y_true, y_pred)
    loss2 = K.categorical_crossentropy(K.ones_like(y_pred) / nb_classes, y_pred)
    return (1 - e) * loss1 + e * loss2


class myVnet(object):
    def __init__(self, img_depth=3, img_rows=img_size, img_cols=img_size, img_channel=1, drop=0.5):
        self.img_depth = img_depth
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channel = img_channel
        self.drop = drop

    def BN_operation(self, input):
        output = keras.layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
                                                               scale=True,
                                                               beta_initializer='zeros', gamma_initializer='ones',
                                                               moving_mean_initializer='zeros',
                                                               moving_variance_initializer='ones',
                                                               beta_regularizer=None,
                                                               gamma_regularizer=None, beta_constraint=None,
                                                               gamma_constraint=None)(input)
        return output

    # def Attention_block(self, input_w, input_x):
    #     wg = Conv2D(64, 1, activation=None, padding='same', kernel_initializer='he_normal')(input_w)
    #     wg = self.BN_operation(wg)
    #     x = Conv2D(64, 1, activation=None, padding='same', kernel_initializer='he_normal')(input_x)
    #     x = self.BN_operation(x)
    #     adds = layers.add([wg, x])
    #     psi = Conv2D(1, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal')(adds)
    #
    #     output = Multiply()([psi, x])
    #     return output

    def Scale_attention_module(self, input_weight, input_map, kernel_num):
        weight = Conv3D(kernel_num, 1, activation=None, padding='same', kernel_initializer='he_normal')(input_weight)
        weight = self.BN_operation(weight)
        map = Conv3D(kernel_num, 1, activation=None, padding='same', kernel_initializer='he_normal')(input_map)
        map = self.BN_operation(map)

        adds = layers.add([weight, map])
        attention = Conv3D(1, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal')(adds)
        # 首先得到基于Attention机制的特征图，对原始特征图目标进行了一次强化
        attention_output = Multiply()([attention, map])

        conv = Conv3D(kernel_num, 1, activation='relu', padding='same', kernel_initializer='he_normal')(attention_output)
        # 然后继续进行scale-attention，继续强化尺度特征
        scale1 = Conv3D(kernel_num, 1, activation='relu', padding='same', kernel_initializer='he_normal')(attention_output)
        scale1 = self.BN_operation(scale1)
        scale1 = Dropout(self.drop)(scale1)
        scale1 = Conv3D(kernel_num, [3, 1, 1], activation='relu', padding='same', kernel_initializer='he_normal')(scale1)
        scale1 = self.BN_operation(scale1)
        scale1 = Dropout(self.drop)(scale1)
        scale1 = layers.add([conv, scale1])
        scale1 = Conv3D(kernel_num, 1, activation='relu', padding='same', kernel_initializer='he_normal')(scale1)

        scale2 = Conv3D(kernel_num, [1, 3, 3], activation='relu', padding='same', kernel_initializer='he_normal')(attention_output)
        scale2 = self.BN_operation(scale2)
        scale2 = Dropout(self.drop)(scale2)
        scale2 = Conv3D(kernel_num, [3, 1, 1], activation='relu', padding='same', kernel_initializer='he_normal')(scale2)
        scale2 = self.BN_operation(scale2)
        scale2 = Dropout(self.drop)(scale2)
        scale2 = layers.add([conv, scale2])
        scale2 = Conv3D(kernel_num, 1, activation='relu', padding='same', kernel_initializer='he_normal')(scale2)

        scale3 = Conv3D(kernel_num, [1, 5, 5], activation='relu', padding='same', kernel_initializer='he_normal')(attention_output)
        scale3 = self.BN_operation(scale3)
        scale3 = Dropout(self.drop)(scale3)
        scale3 = Conv3D(kernel_num, [3, 1, 1], activation='relu', padding='same', kernel_initializer='he_normal')(scale3)
        scale3 = self.BN_operation(scale3)
        scale3 = Dropout(self.drop)(scale3)
        scale3 = layers.add([conv, scale3])
        # res = Conv3D(kernel_num, [3, 1, 1], activation='relu', padding='valid',
        #              kernel_initializer='he_normal')(res)
        scale3 = Conv3D(kernel_num, 1, activation='relu', padding='same', kernel_initializer='he_normal')(scale3)

        # scale1_weight = GlobalAveragePooling3D()(scale1[:, :, :, :])
        # scale2_weight = GlobalAveragePooling3D()(scale2[:, :, :, :])
        # scale3_weight = GlobalAveragePooling3D()(scale3[:, :, :, :])

        scale1_weight = GlobalAveragePooling3D()(scale1)
        scale2_weight = GlobalAveragePooling3D()(scale2)
        scale3_weight = GlobalAveragePooling3D()(scale3)

        scale1 = Multiply()([scale1_weight, scale1])
        scale2 = Multiply()([scale2_weight, scale2])
        scale3 = Multiply()([scale3_weight, scale3])

        return scale1, scale2, scale3

    def encode_layer(self, kernel_num, input):
        scale1, scale2, scale3 = self.Scale_attention_module(input, input, kernel_num)
        output = Concatenate(axis=4)([scale1, scale2, scale3])
        output = Conv3D(kernel_num, 1, activation='relu', padding='same', kernel_initializer='he_normal')(output)
        return tf.convert_to_tensor(output)

    def down_operation(self, kernel_num, kernel_size, input):
        down = Conv3D(kernel_num, kernel_size, strides=[1, 2, 2], activation='relu', padding='same',
                      kernel_initializer='he_normal')(input)
        return down

    def decode_layer(self, kernel_num, kernel_size, input, code_layer):
        deconv = Deconvolution3D(kernel_num, kernel_size, strides=(1, 2, 2), activation='relu', padding='same',
                                 kernel_initializer='he_normal')(input)
        # deconv = Conv3D(kernel_num, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(
        #          UpSampling3D(size=(1, 2, 2))(input))

        merge = Concatenate(axis=4)([deconv, code_layer])
        conv = Conv3D(kernel_num, kernel_size, activation='relu', padding='same',
                      kernel_initializer='he_normal')(merge)
        conv = Dropout(self.drop)(conv)

        res = layers.add([deconv, conv])
        return res

    # V-Net网络
    def get_vnet(self):
        inputs = Input((self.img_depth, self.img_rows, self.img_cols, self.img_channel))

        # 卷积层1
        conv1 = self.encode_layer(32, inputs)
        # 下采样1
        down1 = self.down_operation(64, [1, 3, 3], conv1)

        # 卷积层2
        conv2 = self.encode_layer(64, down1)
        # 下采样2
        down2 = self.down_operation(128, [1, 3, 3], conv2)

        # 卷积层3
        conv3 = self.encode_layer(128, down2)
        # 下采样3
        down3 = self.down_operation(256, [1, 3, 3], conv3)

        # 卷积层4
        conv4 = self.encode_layer(256, down3)
        # 下采样4
        down4 = self.down_operation(512, [1, 3, 3], conv4)

        # 卷积层5
        conv5 = self.encode_layer(512, down4)
        conv5 = Conv3D(512, [3, 1, 1], activation='relu', padding='valid',
                       kernel_initializer='he_normal')(conv5)
        #######################################################################################################################
        # 反卷积6
        deconv6 = Deconvolution3D(256, [1, 3, 3], strides=(1, 2, 2), activation='relu', padding='same',
                                  kernel_initializer='he_normal')(conv5)
        conv4 = Conv3D(256, [3, 1, 1], activation='relu', padding='valid',
                       kernel_initializer='he_normal')(conv4)
        merge6 = Concatenate(axis=4)([deconv6, conv4])
        conv6 = Conv3D(256, [1, 3, 3], activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge6)
        conv6 = Dropout(self.drop)(conv6)
        res6 = layers.add([deconv6, conv6])
        #######################################################################################################################

        #######################################################################################################################
        # 反卷积7
        deconv7 = Deconvolution3D(128, [1, 3, 3], strides=(1, 2, 2), activation='relu', padding='same',
                                  kernel_initializer='he_normal')(res6)
        conv3 = Conv3D(128, [3, 1, 1], activation='relu', padding='valid',
                       kernel_initializer='he_normal')(conv3)
        # conv3 = Conv3D(128, [3, 1, 1], activation='relu', padding='valid',
        #                kernel_initializer='he_normal')(conv3)
        merge7 = Concatenate(axis=4)([deconv7, conv3])
        conv7 = Conv3D(128, [1, 3, 3], activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge7)
        conv7 = Dropout(self.drop)(conv7)
        res7 = layers.add([deconv7, conv7])
        #######################################################################################################################

        #######################################################################################################################
        # 反卷积8
        deconv8 = Deconvolution3D(64, [1, 3, 3], strides=(1, 2, 2), activation='relu', padding='same',
                                  kernel_initializer='he_normal')(res7)
        conv2 = Conv3D(64, [3, 1, 1], activation='relu', padding='valid',
                       kernel_initializer='he_normal')(conv2)
        # conv2 = Conv3D(64, [3, 1, 1], activation='relu', padding='valid',
        #                kernel_initializer='he_normal')(conv2)
        # conv2 = Conv3D(64, [3, 1, 1], activation='relu', padding='valid',
        #                kernel_initializer='he_normal')(conv2)
        merge8 = Concatenate(axis=4)([deconv8, conv2])
        conv8 = Conv3D(64, [1, 3, 3], activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge8)
        conv8 = Dropout(self.drop)(conv8)
        res8 = layers.add([deconv8, conv8])
        #######################################################################################################################

        #######################################################################################################################
        # 反卷积9
        deconv9 = Deconvolution3D(32, [1, 3, 3], strides=(1, 2, 2), activation='relu', padding='same',
                                  kernel_initializer='he_normal')(res8)
        conv1 = Conv3D(32, [3, 1, 1], activation='relu', padding='valid',
                       kernel_initializer='he_normal')(conv1)
        # conv1 = Conv3D(32, [3, 1, 1], activation='relu', padding='valid',
        #                kernel_initializer='he_normal')(conv1)
        # conv1 = Conv3D(32, [3, 1, 1], activation='relu', padding='valid',
        #                kernel_initializer='he_normal')(conv1)
        # conv1 = Conv3D(32, [3, 1, 1], activation='relu', padding='valid',
        #                kernel_initializer='he_normal')(conv1)
        merge9 = Concatenate(axis=4)([deconv9, conv1])
        conv9 = Conv3D(32, [1, 3, 3], activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge9)
        conv9 = Dropout(self.drop)(conv9)
        res9 = layers.add([deconv9, conv9])
        #######################################################################################################################

        conv10 = Conv3D(1, [1, 1, 1], activation='sigmoid')(res9)

        model = Model(inputs=inputs, outputs=conv10)

        # 在这里可以自定义损失函数loss和准确率函数accuracy
        # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        # losses = WeightedBinaryCrossEntropy()
        # model.compile(optimizer=Adam(lr=1e-4), loss=losses.weighted_binary_crossentropy,
        #               metrics=['accuracy', dice_coef])
        # model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=['accuracy',dice_coef])
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy',dice_coef])
        print('model compile')
        return model

    def train(self):
        print("loading data")
        print("loading data done")

        model = self.get_vnet()
        model_save_path = "./model_pre_mean"
        if not os.path.isdir(model_save_path):
            os.makedirs(model_save_path)
        # losses = WeightedBinaryCrossEntropy()
        # model = load_model(model_save_path + '/' + net_name + '_tumour' + str(last_order) + '.hdf5',
        #                    custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss})
        print("got uvnet-scale-attention")
        print(model.summary())  # 显示相关的网络信息

        # 保存的是模型和权重
        model_checkpoint = ModelCheckpoint(model_save_path + '/' + net_name + '_tumour' + str(train_order) + '.hdf5',
                                           monitor='loss',
                                           verbose=1, save_best_only=True)
        print('Fitting model...')

        # 创建一个实例history
        history = LossHistory()
        # 在callbacks中加入history最后才能绘制收敛曲线
        model.fit_generator(
            generator=get_train_batch(train_path_list, label_path_list, train_batch_size, 3, img_size, img_size),
            epochs=epoch, verbose=1,
            steps_per_epoch=count // train_batch_size,
            callbacks=[model_checkpoint, history],
            workers=1)
        # 绘制acc-loss曲线
        history.loss_plot('epoch')
        from keras.utils import plot_model
        # 因为模型结果不能显示，所以保存为图片！！
        model_result_save_path = "./model_result"
        if not os.path.isdir(model_result_save_path):
            os.makedirs(model_result_save_path)
        plot_model(model, to_file=model_result_save_path + '/model_' + net_name + '.png')


if __name__ == '__main__':
    myvnet = myVnet()
    myvnet.train()








