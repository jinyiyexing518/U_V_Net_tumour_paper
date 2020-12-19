""" -*- coding: utf-8 -*-
@ Time: 2020/12/19 12:51
@ author: Zhang Chi
@ e-mail: 220200785@mail.seu.edu.cn
@ file: draw_convergence.py
@ project: U_V_Net_tumour_paper
"""
import matplotlib.pyplot as plt
import numpy as np

# 分别存放所有点的横坐标和纵坐标，一一对应

# 4_3epoch时，效果有所下降
loss_list = [0.2634, 0.1302, 0.0973, 0.0695,
             0.0660, 0.0627, 0.0614, 0.0579, 0.0560]

dice_list = [0.0299, 0.0523, 0.0811, 0.1131,
             0.1221, 0.1269, 0.1457, 0.1444, 0.1629]
x_list = np.arange(1, len(loss_list) + 1)
# 创建图并命名
plt.figure('Line fig')
ax = plt.gca()

# 画连线图，以x_list中的值为横坐标，以y_list中的值为纵坐标
# 参数c指定连线的颜色，linewidth指定连线宽度，alpha指定连线的透明度

# plt.plot(x_list, loss_list, color='green', linestyle='--', marker='>', linewidth=0.5)
plt.plot(x_list, dice_list, color='red', linestyle='--', marker='d', linewidth=0.5)
plt.xlabel('Epoch' + '/' + str(len(x_list)))
# plt.ylabel('Loss')
plt.ylabel('Dice')
# plt.title("Loss Convergence Curve")
plt.title("Dice Convergence Curve")

plt.grid()  # 生成网格
plt.show()



