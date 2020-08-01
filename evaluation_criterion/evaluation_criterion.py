import numpy as np
import cv2 as cv
import os
from surface_distance import metrics


def calPrecision(predict, label):
    """
    计算Precision，predict中，正值的比例
    """
    row, col = predict.shape  # 矩阵的行与列
    TP, TP_NP = 0, 0
    for i in range(row):
        for j in range(col):
            if predict[i][j] == 255:
                TP_NP += 1
                if label[i][j] == 255:
                    TP += 1
    Precision = TP / TP_NP

    return Precision


def calAccuracy(predict, label):
    """
    计算Accuracy，正确点数量
    """
    predict = np.array(predict)
    label = np.array(label)
    row, col = predict.shape  # 矩阵的行与列
    true_point = 0
    for i in range(row):
        for j in range(col):
            if predict[i, j] == label[i, j]:
                true_point += 1
    Accuracy = true_point / (row * col)

    return Accuracy


def calRecall(predict, label):
    """
    计算召回率Recall，label中正确占比
    """
    row, col = predict.shape  # 矩阵的行与列
    TP, TP_FN = 0, 0
    for i in range(row):
        for j in range(col):
            if label[i][j] == 255:
                TP_FN += 1
                if predict[i][j] == 255:
                    TP += 1
    Recall = TP / TP_FN

    return Recall


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


def calJaccard(predict, label):
    """
    Jaccard系数，A∩B / A + B - A∩B
    """
    row, col = predict.shape
    A, B, A_B = 0, 0, 0
    for i in range(row):
        for j in range(col):
            if predict[i][j] == 255:
                A += 1
            if label[i][j] == 255:
                B += 1
            if predict[i][j] == 255 and label[i][j] == 255:
                A_B += 1
    Jaccard = A_B / (A + B - A_B)

    return Jaccard


def calFscore(predict, label):
    """
    F-measure or balanced F-score
    """
    recall = calRecall(predict, label)
    accuracy = calAccuracy(predict, label)
    Fscore = 2 * recall * accuracy / (recall + accuracy)

    return Fscore


def cal_ASSD(predict, label):
    """
    Average Symmetric Surface Distance (ASSD)
    平均表面距离
    :param predict:
    :param label:gt(ground truth)
    :return:
    """
    surface_distances = metrics.compute_surface_distances(
        label, predict, spacing_mm=(1.0, 1.0))
    avg_surf_dist = metrics.compute_average_surface_distance(surface_distances)
    return avg_surf_dist


def cal_hausdorff(predict, label):
    """
     豪斯多夫距离
    :param predict:
    :param label:
    :return:
    """
    surface_distances = metrics.compute_surface_distances(
        label, predict, spacing_mm=(1.0, 1.0))
    hd_dist_95 = metrics.compute_robust_hausdorff(surface_distances, 95)
    return hd_dist_95


def cal_surface_overlap(predict, label):
    """
    Surface overlap 表面重叠度
    :param predict:
    :param label:
    :return:
    """
    surface_distances = metrics.compute_surface_distances(
        label, predict, spacing_mm=(1.0, 1.0))
    surface_overlap = metrics.compute_surface_overlap_at_tolerance(surface_distances, 1)
    return surface_overlap


if __name__ == "__main__":
    current_path = os.getcwd()
    print(current_path)
    file_name = os.listdir(current_path)
    for name in file_name:
        if ".png" not in name:
            file_name.remove(name)
    print(file_name)
    img_num = len(file_name)
    img_list = []
    for i in range(img_num):
        img_path = os.path.join(current_path, file_name[i])
        img = cv.imread(img_path, 0)
        img_list.append(img)
    print(len(img_list))
    predict = img_list[0]
    label = img_list[1]

    print("Precision:{}".format(calPrecision(predict, label)))
    print("Accuracy:{}".format(calAccuracy(predict, label)))
    print("Recall:{}".format(calRecall(predict, label)))
    print("Dice:{}".format(calDice(predict, label)))
    print("Jaccard:{}".format(calJaccard(predict, label)))
    print("Fscore:{}".format(calFscore(predict, label)))

    predict = predict.astype(np.bool)
    label = label.astype(np.bool)
    print("ASSD:{}".format(cal_ASSD(predict, label)))
    # 较常用的metric：Hausdorff distance和Volumetric dice
    print("hausdorff:{}".format(cal_hausdorff(predict, label)))
    print("surface_overlap:{}".format(cal_surface_overlap(predict, label)))
