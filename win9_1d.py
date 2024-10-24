import pickle

import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from CBAM import CBAM
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import os





class Self_Attention(nn.Module):
    def __init__(self, dim, dk, dv,init_weights=True):
        super(Self_Attention, self).__init__()
        self.scale = dk ** -0.5
        self.q = nn.Linear(dim, dk)
        self.k = nn.Linear(dim, dk)
        self.v = nn.Linear(dim, dv)
        if init_weights:
            self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.01)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.fill_(0)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = attn @ v
        return x


class BasicConv1d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0,init_weights=True):
        super(BasicConv1d, self).__init__()
        if init_weights:
            self._initialize_weights()
        self.conv = nn.Conv1d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)  # verify bias false
        self.bn = nn.BatchNorm1d(out_planes,
                                 eps=0.0001,  # value found in tensorflow
                                 momentum=0.1,  # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv1d):
                m.weight.data.normal_(0.0, 0.01)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.normal_(1.0, 0.01)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x = self.relu(x)

        return x

class Inception_A(nn.Module):

    def __init__(self,in_channel):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv1d(in_channel, 64, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv1d(in_channel, 32, kernel_size=1, stride=1),
            BasicConv1d(32, 64, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv1d(in_channel, 32, kernel_size=1, stride=1),
            BasicConv1d(32, 64, kernel_size=3, stride=1, padding=1),
            BasicConv1d(64, 64, kernel_size=3, stride=1, padding=1)
        )

        # self.branch3 = nn.Sequential(
        #     nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
        #     BasicConv2d(in_channel, 32, kernel_size=1, stride=1)
        # )


    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        # x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2), 1)
        return out

class Reduction_A(nn.Module):

    def __init__(self,in_channel):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv1d(in_channel, 64, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv1d(in_channel, 32, kernel_size=1, stride=1),
            BasicConv1d(32, 64, kernel_size=3, stride=1, padding=1),
            BasicConv1d(64, 64, kernel_size=3, stride=2)
        )

        self.branch2 = BasicConv1d(in_channel, 64, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResidualBlock, self).__init__()
        self.branch = nn.Sequential(
            BasicConv1d(in_channel, 64, kernel_size=1, stride=1),
            nn.ReLU(inplace=False),
            BasicConv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            BasicConv1d(64, 64, kernel_size=3, stride=1, padding=1),
        )
        self.relu = nn.ReLU(inplace=False)
    def forward(self, x):
        residual = x
        out = self.branch(x)
        # out = self.cbam(out)
        out =out+residual
        out = self.relu(out)
        return out


class Dense(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate,init_weights=True):
        super(Dense, self).__init__()
        self.dense1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(hidden_size, num_classes)
        # self.dropout2 = nn.Dropout(dropout_rate)
        # self.dense3 = nn.Linear(hidden_size, num_classes)
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.01)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.dense1(x)
        x = self.dropout1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        # x = self.dropout2(x)
        # x = torch.relu(x)
        # x = self.dense3(x)
        return x

class Conv_att(nn.Module):
    def __init__(self,in_channel):#,dim, dk, dv
        super(Conv_att, self).__init__()
        self.conv=BasicConv1d(in_channel,64,kernel_size=1,stride=1)
        self.res_block = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            # ResidualBlock(64),
            # ResidualBlock(64),
            Self_Attention(109*9, 109*9, 128),
            ResidualBlock(64),
            ResidualBlock(64),
            # ResidualBlock(64),
            # ResidualBlock(64),
            Self_Attention(128, 128, 15),
            ResidualBlock(64),
            ResidualBlock(64),
            # ResidualBlock(64),
            # ResidualBlock(64),
            Self_Attention(15, 15, 7),

        )
        self.fc=Dense(64*7,64,2,0.5)
        self.sigmoid = nn.Sigmoid()
        # self.relu=nn.ReLU(inplace=False)

    def forward(self, x):
        x=self.conv(x)
        x = self.res_block(x)
        x = x.view(x.size(0), -1)
        x=self.fc(x)
        return x





