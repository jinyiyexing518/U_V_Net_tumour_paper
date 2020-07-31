import os
import cv2 as cv
import numpy as np


train_slice_path = "./data_raw_slice_liver2/train"
label_slice_path = "./data_raw_slice_liver2/label"


dirs = os.listdir(train_slice_path)
dirs.sort(key=lambda x: int(x))

mean = 0.0
num = 0
j = 0
for dir_name in dirs:
    train_dir_path = os.path.join(train_slice_path, dir_name)
    label_dir_path = os.path.join(label_slice_path, dir_name)

    imgs = os.listdir(train_dir_path)
    imgs.sort(key=lambda x: int(x.split('.')[0]))

    for img in imgs:
        train_img_path = os.path.join(train_dir_path, img)
        label_img_path = os.path.join(label_dir_path, img)

        train_img = cv.imread(train_img_path, 0)
        label_img = cv.imread(label_img_path, 0)

        roi = cv.bitwise_and(train_img, label_img)

        # cv.imshow("roi", roi)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        black_pixel = roi <= 5
        black_num = len(roi[black_pixel])

        mean += roi.mean() * 512 * 512 / (512 * 512 - black_num)
        num += 1
    print("均值=={}".format(mean/num))
    j += 1
    print("完成{}个dir".format(j))

mean /= num
print("均值=={}".format(mean))

