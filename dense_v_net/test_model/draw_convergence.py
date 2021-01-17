import matplotlib.pyplot as plt
import numpy as np

# 分别存放所有点的横坐标和纵坐标，一一对应

# 4_3epoch时，效果有所下降
loss_list = [0.0, 0.0, 0.0,
             0.0, 0.0, 0.0,
             0.0, 0.0, 0.0,
             0.0083, 0.0077, 0.0070, 0.0067, 0.0063, 0.0056, 0.0049, 0.0045,
             0.0042, 0.0041, 0.0038, 0.0036, 0.0034, 0.0033, 0.0033,
             0.0032, 0.0031, 0.0030, 0.0028, 0.0028, 0.0027, 0.0026, 0.0025, 0.0025, 0.0024,
             0.0030, 0.0027, 0.0025, 0.0025, 0.0024, 0.0024, 0.0023, 0.0023, 0.0022, 0.0022
             ]

dice_list = [0.0, 0.0, 0.0,
             0.0, 0.0, 0.0,
             0.0, 0.0, 0.0,
             0.4728, 0.5134, 0.5489, 0.5627, 0.5981, 0.6321, 0.6780, 0.6980,
             0.7171, 0.7267, 0.7455, 0.7665, 0.7731, 0.7849, 0.7870,
             0.7964, 0.8025, 0.8116, 0.8272, 0.8307, 0.8344, 0.8397, 0.8473, 0.8526, 0.8584,
             0.8336, 0.8417, 0.8558, 0.8552, 0.8622, 0.8659, 0.8697, 0.8699, 0.8717, 0.8743
             ]
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



