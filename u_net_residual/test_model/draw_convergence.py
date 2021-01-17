import matplotlib.pyplot as plt
import numpy as np

# 分别存放所有点的横坐标和纵坐标，一一对应

# 4_3epoch时，效果有所下降
loss_list = [0.0401, 0.0379, 0.0341, 0.0336,
             0.0320, 0.0293, 0.0279, 0.0263, 0.0249, 0.0230, 0.0208,  # batch_size = 4
             0.0225, 0.0205, 0.0188, 0.0174, 0.0162, 0.0153, 0.0145, 0.0136, 0.0133, 0.0126,
             0.0122, 0.0117, 0.0113, 0.0109]

dice_list = [0.2996, 0.3198, 0.3655, 0.3868,
             0.4063, 0.4448, 0.4655, 0.4911, 0.5196, 0.5460, 0.5797,  # batch_size = 4
             0.5705, 0.5972, 0.6239, 0.6473, 0.6717, 0.6935, 0.7104, 0.7330, 0.7417, 0.7602,
             0.7729, 0.7851, 0.7935, 0.8035]  # batch_size = 8
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

