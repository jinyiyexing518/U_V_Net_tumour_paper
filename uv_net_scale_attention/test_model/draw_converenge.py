""" -*- coding: utf-8 -*-
@ Time: 2021/1/18 9:37
@ author: Zhang Chi
@ e-mail: 220200785@mail.seu.edu.cn
@ file: draw_converenge.py
@ project: U_V_Net_tumour_paper
"""
import matplotlib.pyplot as plt
import numpy as np

# 分别存放所有点的横坐标和纵坐标，一一对应

# 4_3epoch时，效果有所下降
loss_list = [0.0343, 0.0269, 0.0255]

dice_list = [0.3568, 0.4364, 0.4605]  # batch_size = 8
x_list = np.arange(1, len(loss_list) + 1)
# 创建图并命名
plt.figure('Dice fig')
ax = plt.gca()

# 画连线图，以x_list中的值为横坐标，以y_list中的值为纵坐标
# 参数c指定连线的颜色，linewidth指定连线宽度，alpha指定连线的透明度

plt.plot(x_list, dice_list, color='red', linestyle='--', marker='d', linewidth=0.5)
plt.xlabel('Epoch' + '/' + str(len(x_list)))
plt.ylabel('Dice')
plt.title("Dice Convergence Curve")
plt.grid()  # 生成网格

plt.figure('Loss fig')
plt.plot(x_list, loss_list, color='green', linestyle='--', marker='>', linewidth=0.5)
plt.ylabel('Loss')
plt.title("Loss Convergence Curve")
plt.grid()  # 生成网格

plt.show()


