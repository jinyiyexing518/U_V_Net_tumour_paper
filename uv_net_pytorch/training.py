import torch.backends.cudnn as cudnn
from u_net_pytorch.net.attention_unet import *
from u_net_pytorch.dataloader import *
from torch.autograd import Variable
import time
import os
from torch import optim
import torch


def calDice(y_pred, y_true):
    smooth = 1.
    y_true_f = y_true.ravel()
    y_pred_f = y_pred.ravel()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f * y_true_f) + np.sum(y_pred_f * y_pred_f) + smooth)


def dice_coef_loss(y_pred, y_true):
    return 1. - calDice(y_pred, y_true)


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


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)

    if classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)


def train(train_loader, cfg):
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    """这里修改cpu，gpu版本"""
    # net = Unet().to(device)
    # net = Unet(1, 1)
    net = AttU_Net(1, 1)
    # if cfg.load_checkpoint:
    #     state_dict = torch.load(cfg.checkpoint_dir)
    #     params=state_dict["model_state_dict"]
    #     # for param_tensor in params:  # 打印参数信息
    #     #     print(param_tensor, "\t", params[param_tensor].size())
    #     net.load_state_dict(params)
    #     print(state_dict['epoch'])
    # else:

    for param in net.parameters():
        param.requires_grad = True

    net.apply(weights_init_xavier)  # 权值初始化

    cudnn.benchmark = True

    """这里修改cpu，版本"""
    # criterion_mse = nn.CrossEntropyLoss().to(cfg.device)
    # criterion_mse = nn.CrossEntropyLoss()
    criterion_mse = torch.nn.BCELoss()

    # 优化函数，优化器
    # SGD优化器
    # optimizer = optim.SGD(net.parameters(), lr=cfg.lr, momentum=0.8, weight_decay=0.001)
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
    # 调整学习率，lr = lr * (gamma **(epoch/n_steps)) 每n_steps个epochs调整一次学习率
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)
    k = 0

    for idx_epoch in range(cfg.n_epochs):
        iter_loss_accum = 0.0
        loss_list = []
        min_loss = 5.0
        i = 0
        for index, data in enumerate(train_loader):
            k += 1
            time_start = time.time()
            """这里修改cpu，gpu版本"""
            # train = data[0].to(cfg.device)
            train = Variable(data[0])
            # label = data[1].to(cfg.device)
            label = Variable(data[1])
            predict_label = net(train)
            entropy_loss = criterion_mse(predict_label.reshape(label.shape[0], label.shape[2], label.shape[3]),
                                 label.reshape(label.shape[0], label.shape[2], label.shape[3]))

            pred_label = predict_label.cpu().detach().numpy()
            gt_label = label.cpu().detach().numpy()
            dice = calDice(pred_label, gt_label)
            dice_loss = dice_coef_loss(pred_label, gt_label)
            # 混合损失函数，训练初期，损失函数由交叉熵主导，后期的时候，更关注于分割边缘细节的dice-loss占据主导地位
            loss = (1.0 - idx_epoch/cfg.n_epochs) * entropy_loss + (0.1 * idx_epoch/cfg.n_epochs) * dice_loss

            iter_loss_accum += loss.data
            loss_list.append(loss.data)

            # backward()实现了模型的后向传播中的自动梯度计算
            loss.backward()
            optimizer.step()
            # 直接调用 optimizer.zero_grad()完成对模型参数梯度的归零
            optimizer.zero_grad()
            i += 1

            pred_label = np.reshape(pred_label, (400, 400))
            pred_label *= 255
            pred_label = pred_label.astype('uint8')
            pred_label[pred_label > 50] = 255
            pred_label[pred_label <= 50] = 0

            gt_label = np.reshape(gt_label, (400, 400))
            gt_label *= 255
            gt_label = gt_label.astype('uint8')

            accuracy = calAccuracy(pred_label, gt_label)

            print("Epoch: %d, Iteration: %d, loss: %.6f, accuracy: %.6f, dice: %.6f, time: %f" % (idx_epoch, i,
                                                                                      float(loss.data.cpu()),
                                                                                      float(accuracy),
                                                                                      float(dice),
                                                                                      time.time()-time_start))

        if np.mean(loss_list) < min_loss:
            min_loss = np.mean(loss_list)
            if not os.path.isdir("./model_pre_mean"):
                os.makedirs("./model_pre_mean")
            torch.save("./model_pre_mean/model.pkl")
            print("model saved! \n")
        else:
            print("model not saved! \n")
        # 根据epoch进行学习率衰减
        if idx_epoch % 5 == 0:
            scheduler.step()










