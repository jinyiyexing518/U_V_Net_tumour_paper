import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from u_net.test_model.plot_test_dice_histogram import plot_test_dice_histogram
# from vnet_25D.vnet11_1.test_model.vnet_test import dir_num
from evaluation_criterion.eval_criterion import calPrecision, calAccuracy, calRecall, calFscore, calJaccard, \
    cal_ASSD, cal_hausdorff, cal_surface_overlap


# 计算DICE系数
def calDice(binary_GT,binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    DSI_s, DSI_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                DSI_s += 1
            if binary_GT[i][j] == 255:
                DSI_t += 1
            if binary_R[i][j] == 255:
                DSI_t += 1
    Dice = 2*DSI_s/DSI_t
    return Dice


from u_net.test_model.unet_test import dir_num, model_num, model
# dir_num = 130
# model = "liver"

imgs_size = 400

predict_npy = "./predict_npy_pre_mean/predict" + str(dir_num) + "_" + model\
              + "_epoch" + model_num + ".npy"

# 标签
label_path = "../dataset/unet_1_1_test_npy/label/" + str(dir_num) + "/" + str(dir_num) + ".npy"
# label_path = "../dataset/vnet_3_1_test_npy/raw_label/" + str(dir_num) + "/" + str(dir_num) + ".npy"
train_path = "../dataset/unet_1_1_test_npy/train/" + str(dir_num) + "/" + str(dir_num) + ".npy"

npy_data = np.load(predict_npy)
npy_label = np.load(label_path)
npy_train = np.load(train_path)
print("共预测了{}组数据".format(len(npy_data)))
dice_list = []
precision_list = []
accuracy_list = []
fscore_list = []
jaccard_list = []
recall_list = []
assd_list = []
haus_list = []
surface_list = []
for i in range(len(npy_data)):
    img = npy_data[i]
    label = npy_label[i]
    #######################
    # raw_label
    # label = npy_label[i + 1]
    # label = label[0: 400, 50: 450]
    #######################
    train = npy_train[i]
    label = np.reshape(label, (imgs_size, imgs_size))
    train = np.reshape(train, (imgs_size, imgs_size))

    img = np.reshape(img, (imgs_size, imgs_size))
    img *= 255
    img = img.astype('uint8')

    # 这里应该使用copy，否则会指向同一存储单元，原始img改变，此值随之改变
    origin = img.copy()

    img[img > 50] = 255
    img[img <= 50] = 0

    ################################################
    # 开操作和闭操作
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    # img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    # img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    ################################################

    # 计算原始label像素点数
    white_pixel = label == 255
    pixel_num = len(label[white_pixel])

    # 计算评价指标
    dice = calDice(label, img)
    Precision = calPrecision(img, label)
    Recall = calRecall(img, label)
    Fscore = calFscore(img, label)
    Jaccard = calJaccard(img, label)
    Accuracy = calAccuracy(img, label)
    # 较常用的metric：Hausdorff distance和Volumetric dice
    ASSD = cal_ASSD(img, label)
    hausdorff = cal_hausdorff(img, label)
    surface_overlap = cal_surface_overlap(img, label)

    print("第{}组数据---dice:{} precison:{} recall:{} fscore:{} jaccard:{} accuracy:{} "
          "ASSD:{} hausdorff:{} surface_overlap:{}"
          .format(i+1, dice, Precision, Recall, Fscore, Jaccard, Accuracy,
                  ASSD, hausdorff, surface_overlap), pixel_num)
    dice_list.append(dice*100)
    precision_list.append(Precision*100)
    recall_list.append(Recall*100)
    fscore_list.append(Fscore*100)
    jaccard_list.append(Jaccard*100)
    accuracy_list.append(Accuracy*100)
    assd_list.append(ASSD*100)
    haus_list.append(hausdorff*100)
    surface_list.append(surface_overlap*100)

    # 查看dice系数较小的分割结果
    # 会发现，dice系数较小的分割结果，往往都是肝脏占比比较小的图像
    # if int(dice*100) >= 98:
    # if int(dice*100) < 90 or int(dice*100) >= 99:
    # if i % 10 == 0:
    # if i >= 80:
    # if i >= 50:
    if False:
    # if True:
        plt.figure(1)
        plt.subplot(1, 4, 1)
        # plt.title('train---No.' + str(i))
        plt.imshow(train, cmap='gray')

        plt.subplot(1, 4, 2)
        plt.title('dice---' + str(float("%0.3f" % dice)))
        plt.imshow(img, cmap='gray')

        plt.subplot(1, 4, 3)
        # plt.title('label---' + str(pixel_num))
        # print(pixel_num)
        plt.imshow(label, cmap='gray')
        plt.subplot(1, 4, 4)
        plt.title('origin')
        plt.imshow(origin, cmap='gray')

        # plt.savefig("predict_" + str(i) + ".png")

        # plt.savefig()

        # 显示灰度直方图
        # plt.figure(2)
        # predict_hist = cv.equalizeHist(origin)
        # plt.hist(predict_hist.ravel(), 256, [1, 256])

        plt.show()
    # else:
    #     pass

plot_test_dice_histogram(dice_list)
print("测试数据的平均dice值：{}".format(np.mean(dice_list)))
print("测试数据的平均precision值：{}".format(np.mean(precision_list)))
print("测试数据的平均recall值：{}".format(np.mean(recall_list)))
print("测试数据的平均fscore值：{}".format(np.mean(fscore_list)))
print("测试数据的平均jaccard值：{}".format(np.mean(jaccard_list)))
print("测试数据的平均accuracy值：{}".format(np.mean(accuracy_list)))
print("ASSD:{}, full marks:{}".format(np.mean(assd_list), (0.0, 0.0)))
print("hausdorff:{}, full marks:{}".format(np.mean(haus_list), 0.0))
print("surface_overlap:{}, full marks:{}".format(np.mean(surface_list), (1.0, 1.0)))

