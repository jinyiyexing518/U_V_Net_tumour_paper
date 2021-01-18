这里的版本适合unet 2D的网络
所有在2D的unet基础上的网络都可以使用
只要修改model部分即可

main.py       主函数         需要修改一些超参数
（model）.py      网络程序       需要修改网络模型
training.py   训练程序       需要修改训练的方式
dataloader.py 导入数据程序   需要修改数据预处理