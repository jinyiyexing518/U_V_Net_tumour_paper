""" -*- coding: utf-8 -*-
@ Time: 2021/1/18 13:59
@ author: Zhang Chi
@ e-mail: 220200785@mail.seu.edu.cn
@ file: attention_unet.py
@ project: U_V_Net_tumour_paper
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class AttU_Net(nn.Module):
    def __init__(self, in_ch=1, output_ch=1):
        super(AttU_Net, self).__init__()

        # encoder
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = DoubleConv(512, 1024)

        # decoder
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.Att6 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.conv6 = DoubleConv(1024, 512)

        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.Att7 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.conv7 = DoubleConv(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.Att8 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.conv8 = DoubleConv(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.Att9 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.conv9 = DoubleConv(128, 64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)

        x2 = self.pool1(x1)
        x2 = self.conv2(x2)

        x3 = self.pool2(x2)
        x3 = self.conv3(x3)

        x4 = self.pool3(x3)
        x4 = self.conv4(x4)

        x5 = self.pool4(x4)
        x5 = self.conv5(x5)

        # decoding + concat path
        d5 = self.up6(x5)
        x4 = self.Att6(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.conv6(d5)

        d4 = self.up7(d5)
        x3 = self.Att7(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.conv7(d4)

        d3 = self.up8(d4)
        x2 = self.Att8(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.conv8(d3)

        d2 = self.up9(d3)
        x1 = self.Att9(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.conv9(d2)

        d1 = self.Conv_1x1(d2)

        out = nn.Sigmoid()(d1)

        return out









