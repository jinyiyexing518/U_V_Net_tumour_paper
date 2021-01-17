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
# 这是map=32的模型收敛情况，收敛较慢，所以重新训练map=64的模型
# loss_list = [0.2634, 0.1302, 0.0973, 0.0695,
#              0.0660, 0.0627, 0.0614, 0.0579, 0.0560,
#              0.0548, 0.0528, 0.0513,
#              0.0540, 0.0532, 0.0521,
#              0.0500, 0.0487, 0.0464, 0.0451, 0.0446, 0.0446,
#              0.0434, 0.0423, 0.0406]
#
# dice_list = [0.0299, 0.0523, 0.0811, 0.1131,
#              0.1221, 0.1269, 0.1457, 0.1444, 0.1629,
#              0.1548, 0.1656, 0.1824,
#              0.15016, 0.1567, 0.1726,
#              0.1824, 0.1965, 0.2227, 0.2347, 0.2449, 0.2458,
#              0.2530, 0.2675, 0.2803]
# loss_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ]
#
# dice_list = []

loss_list = []
dice_list = []

x_list = np.arange(1, len(loss_list) + 1)


# 画连线图，以x_list中的值为横坐标，以y_list中的值为纵坐标
# 参数c指定连线的颜色，linewidth指定连线宽度，alpha指定连线的透明度

# 创建图并命名
plt.figure('Dice fig')
ax1 = plt.gca()
plt.plot(x_list, dice_list, color='red', linestyle='--', marker='d', linewidth=0.5)
plt.xlabel('Epoch' + '/' + str(len(x_list)))
plt.ylabel('Dice')
plt.title("Dice Convergence Curve")
plt.grid()  # 生成网格

plt.figure('Loss fig')
ax2 = plt.gca()
plt.plot(x_list, loss_list, color='green', linestyle='--', marker='>', linewidth=0.5)
plt.xlabel('Epoch' + '/' + str(len(x_list)))
plt.ylabel('Loss')
plt.title("Loss Convergence Curve")
plt.grid()  # 生成网格

plt.show()



