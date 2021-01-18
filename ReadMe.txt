本工程文件夹，为了paper，做对比实验
dataset中的数据，预处理结果可以直接从V_Net_tumour_processing_data工程文件夹中获取

vnet_25D_para:model3 86.87
vnet_25D_processing的模型是：88.92（在另一个工程文件夹中）

现在已经放入version-pytorch文件夹中
u_net_pytorch：搭建的pytorch框架下，基于2D Unet的网络以及训练框架，包含混合损失函数
uv_net_pytorch：搭建的基于pytorch框架下，基于uvnet的网络以及训练框架
------------------------------------------------------------------------------------
以下是新的文件夹分布介绍-----------
------------------------------------------------------------------------------------
some_method：一些功能实现的子文件夹，用于评估，展示，修改等功能，主要用于数据可视化，写论文使用
version_keras：初期用keras框架搭建的网络
version_pytorch：使用pytorch框架搭建的网络，更容易修改网络，加入自己思考的模块
version_tensorflow：使用tensorflow框架搭建的网络



