""" -*- coding: utf-8 -*-
@ Time: 2020/12/18 13:47
@ author: Zhang Chi
@ e-mail: 220200785@mail.seu.edu.cn
@ file: unet_attention_tumour.py
@ project: U_V_Net_tumour_paper
"""
import sys
import codecs
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
import keras
from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Multiply
from keras.optimizers import *
from keras import layers
from keras.layers.core import Lambda
from keras.layers import Concatenate

from keras import backend as K

from keras.callbacks import ModelCheckpoint
from u_net_double_plus.net.fit_generator import get_path_list, get_train_batch
import matplotlib.pyplot as plt

# 每次训练模型之前，需要修改的三个地方，训练数据地址、保存模型地址、保存训练曲线地址

net_name = "unet_attention"
train_order = 1
last_order = 0

train_batch_size = 1
epoch = 5
img_size = 400

# data_train_path = "./data_dir_png/train"
# data_label_path = "./data_dir_png/label"
data_train_path = "D:/pycharm_project/U_V_Net_tumour_paper/u_net/dataset/data_cv_clip/train"
data_label_path = "D:/pycharm_project/U_V_Net_tumour_paper/u_net/dataset/data_cv_clip/label"

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
        plt.figure(1)
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train dice')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('dice')
        plt.legend(loc="best")
        #       plt.savefig('./curve_figure/unet_pure_liver_raw_0_129_entropy_dice_curve.png')
        curve_figure_save_path = "./curve_figure"
        plt.savefig(curve_figure_save_path + '/' + net_name + '_tumour_dice' + str(train_order) + '.png')

        plt.figure(2)
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="best")
        #       plt.savefig('./curve_figure/unet_pure_liver_raw_0_129_entropy_loss_curve.png')
        plt.savefig(curve_figure_save_path + '/' + net_name + 'unet''_tumour_loss' + str(train_order) + '.png')
        plt.show()


class WeightedBinaryCrossEntropy(object):

    def __init__(self, pos_ratio=0.7):
        neg_ratio = 1. - pos_ratio
        self.pos_ratio = tf.constant(pos_ratio, tf.float32)
        self.weights = tf.constant(neg_ratio / pos_ratio, tf.float32)
        self.__name__ = "weighted_binary_crossentropy({0})".format(pos_ratio)

    def __call__(self, y_true, y_pred):
        return self.weighted_binary_crossentropy(y_true, y_pred)

    def weighted_binary_crossentropy(self, y_true, y_pred):
        # Transform to logits
        epsilon = tf.convert_to_tensor(K.common._EPSILON, y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        y_pred = tf.log(y_pred / (1 - y_pred))

        cost = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, self.weights)
        return K.mean(cost * self.pos_ratio, axis=-1)


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


class myUnet(object):
    def __init__(self, img_rows=img_size, img_cols=img_size):
        self.img_rows = img_rows
        self.img_cols = img_cols

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

    def Attention_block(self, input_w, input_x):
        wg = Conv2D(64, 1, activation=None, padding='same', kernel_initializer='he_normal')(input_w)
        wg = self.BN_operation(wg)
        x = Conv2D(64, 1, activation=None, padding='same', kernel_initializer='he_normal')(input_x)
        x = self.BN_operation(x)
        adds = layers.add([wg, x])
        # 这里的激活函数都需要改成sigmoid，之前训练的网络都忘记修改了
        # psi = Conv2D(1, 1, activation='relu', padding='same', kernel_initializer='he_normal')(adds)
        psi = Conv2D(1, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal')(adds)
        # def mul(inputs):
        #     psi, x = inputs
        #     return psi * x
        # output = Lambda(mul)([psi, x])

        output = Multiply()([psi, x])
        return output

    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 1))

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        # BN
        # pool1 = self.BN_operation(pool1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        # BN
        # pool2 = self.BN_operation(pool2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        # BN
        # pool3 = self.BN_operation(pool3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        # BN
        # pool4 = self.BN_operation(pool4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)
        # BN
        # drop5 = self.BN_operation(drop5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = self.Attention_block(drop4, up6)
        # merge6 = Concatenate(axis=3)([drop4, up6])
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = self.Attention_block(conv3, up7)
        # merge7 = Concatenate(axis=3)([conv3, up7])
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = self.Attention_block(conv2, up8)
        # merge8 = Concatenate(axis=3)([conv2, up8])
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = self.Attention_block(conv1, up9)
        # merge9 = Concatenate(axis=3)([conv1, up9])
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=conv10)

        # 在这里可以自定义损失函数loss和准确率函数accuracy
        # losses = WeightedBinaryCrossEntropy()
        # model.compile(optimizer=Adam(lr=1e-4), loss=losses.weighted_binary_crossentropy,
        # metrics=['accuracy',dice_coef])
        # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy', dice_coef])
        print('model compile')
        return model

    def train(self):
        print("loading data")

        print("loading data done")
        model = self.get_unet()
        model_save_path = "../model_pre_mean"
        if not os.path.isdir(model_save_path):
            os.makedirs(model_save_path)
        # losses = WeightedBinaryCrossEntropy()
        # model = load_model(model_save_path + '/' + net_name + '_tumour' + str(last_order) + '.hdf5',
        #                    custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss})
        print("got unet")
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
            generator=get_train_batch(train_path_list, label_path_list, train_batch_size, img_size, img_size),
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
    myunet = myUnet()
    myunet.train()

















