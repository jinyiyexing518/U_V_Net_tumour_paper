import matplotlib.pyplot as plt
import numpy as np

# 分别存放所有点的横坐标和纵坐标，一一对应

# 4_3epoch时，效果有所下降
loss_list = [0.0262, 0.0213, 0.0185, 0.0168, 0.0150, 0.0132, 0.0124, 0.0112, 0.0105, 0.0096,
             0.0092, 0.0087, 0.0083, 0.0079, 0.0076, 0.0073, 0.0070, 0.0068, 0.0065, 0.0062,
             0.0060, 0.0059, 0.0056, 0.0055, 0.0053, 0.0052, 0.0051, 0.0049, 0.0048, 0.0047,
             0.0046, 0.0045, 0.0044, 0.0043, 0.0042, 0.0041, 0.0040, 0.0040, 0.0039, 0.0039,
             0.0038, 0.0038, 0.0037, 0.0037, 0.0036, 0.0036, 0.0035, 0.0035, 0.0035, 0.0034]
dice_list = [0.4570, 0.5371, 0.5845, 0.6193, 0.6562, 0.6899, 0.7109, 0.7344, 0.7493, 0.7712,
             0.7818, 0.7956, 0.8059, 0.8151, 0.8229, 0.8318, 0.8385, 0.8452, 0.8486, 0.8586,
             0.8637, 0.8673, 0.8750, 0.8772, 0.8836, 0.8856, 0.8900, 0.8942, 0.8959, 0.8992,
             0.9010, 0.9033, 0.9050, 0.9091, 0.9115, 0.9135, 0.9133, 0.9143, 0.9165, 0.9197,
             0.9202, 0.9221, 0.9234, 0.9242, 0.9253, 0.9258, 0.9282, 0.9289, 0.9284, 0.9317]
x_list = np.arange(1, len(loss_list) + 1)
# 创建图并命名
plt.figure('Dice fig')
# ax = plt.gca()
# 画连线图，以x_list中的值为横坐标，以y_list中的值为纵坐标
# 参数c指定连线的颜色，linewidth指定连线宽度，alpha指定连线的透明度
plt.plot(x_list, dice_list, color='red', linestyle='--', marker='d', linewidth=0.5)
plt.xlabel('Epoch' + '/' + str(len(x_list)))
plt.ylabel('Dice')
plt.title("Dice Convergence Curve")
plt.grid()  # 生成网格

plt.figure('Loss fig')
# ax2 = plt.gca()
# 画连线图，以x_list中的值为横坐标，以y_list中的值为纵坐标
# 参数c指定连线的颜色，linewidth指定连线宽度，alpha指定连线的透明度
plt.plot(x_list, loss_list, color='green', linestyle='--', marker='>', linewidth=0.5)
plt.xlabel('Epoch' + '/' + str(len(x_list)))
plt.ylabel('Loss')
plt.title("Loss Convergence Curve")
plt.grid()  # 生成网格

plt.show()
