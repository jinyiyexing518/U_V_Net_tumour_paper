import sys
import codecs
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
import keras
from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import *
from keras import layers
from keras.layers import Concatenate
from keras import activations
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from refine_net.net.fit_generator import get_path_list, get_train_batch
import matplotlib.pyplot as plt

# 每次训练模型之前，需要修改的三个地方，训练数据地址、保存模型地址、保存训练曲线地址

train_batch_size = 1
epoch = 10
img_size = 400

# data_train_path = "../../u_net/dataset/data_dir_png/train"
# data_label_path = "../../u_net/dataset/data_dir_png/label"
data_train_path = "../../u_net/dataset/data_cv_clip/train"
data_label_path = "../../u_net/dataset/data_cv_clip/label"


train_path_list, label_path_list, count = get_path_list(data_train_path, data_label_path)


# 写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
   def on_train_begin(self, logs={}):
       self.losses = {'batch': [], 'epoch':[]}
       self.accuracy = {'batch': [], 'epoch':[]}
       self.val_loss = {'batch': [], 'epoch':[]}
       self.val_acc = {'batch': [], 'epoch':[]}

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
       plt.savefig('./curve_figure/refine_net_tumour_dice1.png')
       
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
       plt.savefig('./curve_figure/refine_net_tumour_loss1.png')
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


def tensor_sum(x):
    return K.sum(x)


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
        bn = keras.layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
                                                           scale=True,
                                                           beta_initializer='zeros', gamma_initializer='ones',
                                                           moving_mean_initializer='zeros',
                                                           moving_variance_initializer='ones',
                                                           beta_regularizer=None,
                                                           gamma_regularizer=None, beta_constraint=None,
                                                           gamma_constraint=None)(input)
        return bn

    def res_conv_unit(self, kernel_num, input):
        conv1 = Conv2D(kernel_num, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(input)
        conv2 = Conv2D(kernel_num, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv1)
        conv3 = Conv2D(kernel_num, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv2)
        res = layers.add([conv3, conv1])
        conv = Conv2D(kernel_num, 1, activation='relu', padding='same',
                      kernel_initializer='he_normal')(res)

        return conv

    def chained_residual_pool(self, kernel_num, input):
        actives = Conv2D(kernel_num, 3, activation='relu', padding='same',
                         kernel_initializer='he_normal')(input)
        pool1 = Conv2D(kernel_num, 5, activation='relu', padding='same',
                       kernel_initializer='he_normal')(actives)
        pool1 = Conv2D(kernel_num, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool1)
        add1 = layers.add([actives, pool1])

        pool2 = Conv2D(kernel_num, 5, activation='relu', padding='same',
                       kernel_initializer='he_normal')(add1)
        pool2 = Conv2D(kernel_num, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool2)
        add2 = layers.add([add1, pool2])

        pool3 = Conv2D(kernel_num, 5, activation='relu', padding='same',
                       kernel_initializer='he_normal')(add2)
        pool3 = Conv2D(kernel_num, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool3)
        add3 = layers.add([add2, pool3])

        return add3

    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 1))
        lay1 = self.res_conv_unit(64, inputs)
        pool1 = MaxPooling2D(pool_size=(2, 2))(lay1)
        pool1 = self.BN_operation(pool1)

        lay2 = self.res_conv_unit(128, pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(lay2)
        pool2 = self.BN_operation(pool2)

        lay3 = self.res_conv_unit(256, pool2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(lay3)
        pool3 = self.BN_operation(pool3)

        lay4 = self.res_conv_unit(128, pool3)
        pool4 = MaxPooling2D(pool_size=(2, 2))(lay4)
        pool4 = self.BN_operation(pool4)
        # Residual convlution unit
        # 第四层
        res4 = self.res_conv_unit(512, pool4)
        res4 = self.res_conv_unit(512, res4)
        res4 = Conv2D(512, [3, 3], activation='relu', padding='same',
                      kernel_initializer='he_normal')(res4)
        up4 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(res4))
        chained_pool4 = self.chained_residual_pool(512, up4)
        chained_pool4 = self.res_conv_unit(512, chained_pool4)
        # 第三层
        res3 = self.res_conv_unit(256, pool3)
        res3 = self.res_conv_unit(256, res3)
        res3 = Conv2D(256, 3, activation='relu', padding='same',
                      kernel_initializer='he_normal')(res3)
        up3 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(res3))
        chained_pool4 = Conv2D(256, 3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(chained_pool4)
        chained_pool4 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                      UpSampling2D(size=(2, 2))(chained_pool4))
        up34 = layers.add([up3, chained_pool4])
        chained_pool3 = self.chained_residual_pool(256, up34)
        chained_pool3 = self.res_conv_unit(256, chained_pool3)
        # 第二层
        res2 = self.res_conv_unit(256, pool2)
        res2 = self.res_conv_unit(256, res2)
        res2 = Conv2D(128, 3, activation='relu', padding='same',
                      kernel_initializer='he_normal')(res2)
        up2 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(res2))
        chained_pool3 = Conv2D(128, 3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(chained_pool3)
        chained_pool3 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                      UpSampling2D(size=(2, 2))(chained_pool3))
        up234 = layers.add([up2, chained_pool3])
        chained_pool2 = self.chained_residual_pool(128, up234)
        chained_pool2 = self.res_conv_unit(128, chained_pool2)
        # 第一层
        res1 = self.res_conv_unit(256, pool1)
        res1 = self.res_conv_unit(256, res1)
        res1 = Conv2D(64, 3, activation='relu', padding='same',
                      kernel_initializer='he_normal')(res1)
        up1 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(res1))
        chained_pool2 = Conv2D(64, 3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(chained_pool2)
        chained_pool2 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                      UpSampling2D(size=(2, 2))(chained_pool2))
        up1234 = layers.add([up1, chained_pool2])
        chained_pool1 = self.chained_residual_pool(64, up1234)
        chained_pool1 = self.res_conv_unit(64, chained_pool1)

        conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(chained_pool1)
        conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=conv10)

        # 在这里可以自定义损失函数loss和准确率函数accuracy
        # losses = WeightedBinaryCrossEntropy()
        # model.compile(optimizer=Adam(lr=1e-4), loss=losses.weighted_binary_crossentropy,
        #               metrics=['accuracy', dice_coef])
        # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy', dice_coef])
        print('model compile')
        return model

    def train(self):
        print("loading data")

        print("loading data done")
        model = self.get_unet()
        # losses = WeightedBinaryCrossEntropy()
        # model = load_model('./model_pre_mean/refine_net_tumour1.hdf5', custom_objects={'dice_coef': dice_coef,
        #                                                                          'dice_coef_loss': dice_coef_loss})
        print("got unet")
        # print(model.summary())  # 显示相关的网络信息

        # 保存的是模型和权重
        model_checkpoint = ModelCheckpoint('./model_pre_mean/refine_net_tumour1.hdf5',
                                           monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')

        # 创建一个实例history
        history = LossHistory()
        # 在callbacks中加入history最后才能绘制收敛曲线
        model.fit_generator(
            generator=get_train_batch(train_path_list, label_path_list, train_batch_size, img_size, img_size),
            epochs=epoch, verbose=1,
            steps_per_epoch=count//train_batch_size,
            callbacks=[model_checkpoint, history],
            workers=1)
        # 绘制acc-loss曲线
        history.loss_plot('epoch')
        from keras.utils import plot_model
        # 因为模型结果不能显示，所以保存为图片！！
        plot_model(model, to_file='model_refine_net_tumour.png')


if __name__ == '__main__':
    myunet = myUnet()
    myunet.train()








