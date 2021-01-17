import numpy as np
import cv2 as cv
import os

data_train_path = "./deform/train"
data_label_path = "./deform/label"


def get_path_list(data_train_path, data_label_path):
    dirs = os.listdir(data_train_path)
    dirs.sort(key=lambda x: int(x))
    
    count = 0
    train_path_list = []
    label_path_list = []
    for dir in dirs:
        train_dir_path = os.path.join(data_train_path, dir)
        label_dir_path = os.path.join(data_label_path, dir)
        imgs = os.listdir(train_dir_path)
        imgs.sort(key=lambda x: int(x.split('.')[0]))
        count += len(imgs)
        for img in imgs:
            train_img_path = os.path.join(train_dir_path, img)
            label_img_path = os.path.join(label_dir_path, img)
            train_path_list.append(train_img_path)
            label_path_list.append(label_img_path)
    print("共有{}组训练数据".format(count))
    return train_path_list, label_path_list, count


def get_train_img(paths, img_rows, img_cols):
    """
    参数：
        paths：要读取的图片路径列表
        img_rows:图片行
        img_cols:图片列
        color_type:图片颜色通道
    返回:
        imgs: 图片数组
    """
    # Load as grayscale
    imgs = []
    for path in paths:
        img = cv.imread(path, 0)
        # Reduce size
        resized = np.reshape(img, (img_rows, img_cols, 1))
        resized = resized.astype('float32')
        resized /= 255
#        mean = resized.mean(axis=0)
#        resized -= mean
        imgs.append(resized)
    imgs = np.array(imgs)
    return imgs


def get_label_img(paths, img_rows, img_cols):
    """
    参数：
        paths：要读取的图片路径列表
        img_rows:图片行
        img_cols:图片列
        color_type:图片颜色通道
    返回:
        imgs: 图片数组
    """
    # Load as grayscale
    imgs = []
    for path in paths:
        img = cv.imread(path, 0)
        # Reduce size
        resized = np.reshape(img, (img_cols, img_rows, 1))
        resized = resized.astype('float32')
        resized /= 255
        imgs.append(resized)
    imgs = np.array(imgs)
    return imgs


def get_train_batch(train, label, batch_size, img_w, img_h):
    """
    参数：
        X_train：所有图片路径列表
        y_train: 所有图片对应的标签列表
        batch_size:批次
        img_w:图片宽
        img_h:图片高
        color_type:图片类型
        is_argumentation:是否需要数据增强
    返回:
        一个generator，x: 获取的批次图片 y: 获取的图片对应的标签
    """
    while 1:
        for i in range(0, len(train), batch_size):
            x = get_train_img(train[i:i+batch_size], img_w, img_h)
            y = get_label_img(label[i:i+batch_size], img_w, img_h)
            # 最重要的就是这个yield，它代表返回，返回以后循环还是会继续，然后再返回。就比如有一个机器一直在作累加运算，但是会把每次累加中间结果告诉你一样，直到把所有数加完
            yield(np.array(x), np.array(y))


if __name__ == "__main__":
    train_path_list, label_path_list = get_path_list(data_train_path, data_label_path)

