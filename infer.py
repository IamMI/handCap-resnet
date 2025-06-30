"""
Load pretrained params and take a inference to test
"""

import os
import argparse

from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append('components/hand_keypoints/')
from utils.model_utils import *
from utils.common_utils import *
from hand_data_iter.datasets import *
from components.hand_keypoints.models.resnet import resnet18,resnet34,resnet50,resnet101
from components.hand_keypoints.models.squeezenet import squeezenet1_1,squeezenet1_0
from components.hand_keypoints.models.shufflenetv2 import ShuffleNetV2
from components.hand_keypoints.models.shufflenet import ShuffleNet
from components.hand_keypoints.models.mobilenetv2 import MobileNetV2
from loss.loss import *
import time
import json
from loguru import logger
import torch.nn as nn


def calc_mes(predicted, target):
    squared_diff = np.square(predicted - target)
    mes = np.mean(squared_diff)
    return mes

def infer(ops):
    try:
        if ops.model == 'resnet_50':
            model_ = resnet50(pretrained = pretrained, num_classes = ops.num_classes,img_size = ops.img_size[0],dropout_factor=ops.dropout)
        elif ops.model == 'resnet_18':
            model_ = resnet18(pretrained = pretrained, num_classes = ops.num_classes,img_size = ops.img_size[0],dropout_factor=ops.dropout)
        elif ops.model == 'resnet_34':
            model_ = resnet34(pretrained = pretrained, num_classes = ops.num_classes,img_size = ops.img_size[0],dropout_factor=ops.dropout)
        elif ops.model == 'resnet_101':
            model_ = resnet101(pretrained = pretrained, num_classes = ops.num_classes,img_size = ops.img_size[0],dropout_factor=ops.dropout)
        elif ops.model == "squeezenet1_0":
            model_ = squeezenet1_0(pretrained=pretrained, num_classes=ops.num_classes,dropout_factor=ops.dropout)
        elif ops.model == "squeezenet1_1":
            model_ = squeezenet1_1(pretrained=pretrained, num_classes=ops.num_classes,dropout_factor=ops.dropout)
        elif ops.model == "shufflenetv2":
            model_ = ShuffleNetV2(ratio=1., num_classes=ops.num_classes, dropout_factor=ops.dropout)
        elif ops.model == "shufflenet":
            model_ = ShuffleNet(num_blocks = [2,4,2], num_classes=ops.num_classes, groups=3, dropout_factor = ops.dropout)
        elif ops.model == "mobilenetv2":
            model_ = MobileNetV2(num_classes=ops.num_classes , dropout_factor = ops.dropout)

        else:
            print(" no support the model")
        print(model_)
        use_cuda = torch.cuda.is_available()

        device = torch.device("cuda" if use_cuda else "cpu")


        # Dataset
        print("Loading Dataset....")
        dataset = LoadImagesAndLabels(ops=ops, img_size=ops.img_size,flag_agu=ops.flag_agu,fix_res = ops.fix_res,vis = False)
        print("handpose done")

        print('len train datasets : %s'%(dataset.__len__()))
        # Dataloader
        dataloader = DataLoader(dataset,
                                batch_size=ops.batch_size,
                                num_workers=ops.num_workers,
                                shuffle=True,
                                pin_memory=False,
                                drop_last = True)
        # 优化器设计
        optimizer_Adam = torch.optim.Adam(model_.parameters(), lr=ops.init_lr, betas=(0.9, 0.99), weight_decay=1e-6)
        # optimizer_SGD = optim.SGD(model_.parameters(), lr=ops.init_lr, momentum=ops.momentum, weight_decay=ops.weight_decay)# 优化器初始化
        optimizer = optimizer_Adam
        # 定义学习率调度器
        # scheduler = lr_scheduler.StepLR(optimizer, 2, gamma=0.1)
        # 加载 finetune 模型
        if os.access(ops.fintune_model, os.F_OK):# checkpoint
            chkpt = torch.load(ops.fintune_model, map_location='cpu')
            print(model_.load_state_dict(chkpt))
            print('load fintune model : {}'.format(ops.fintune_model))
            del chkpt
        model_ = model_.to(device)
        print('/**********************************************/')
        # 损失函数
        if ops.loss_define != 'wing_loss':
            criterion = nn.MSELoss(reduce=True, reduction='mean')

        step = 0
        idx = 0

        # 变量初始化
        best_loss = np.inf
        loss_mean = 0. # 损失均值
        loss_idx = 0. # 损失计算计数器
        flag_change_lr_cnt = 0 # 学习率更新计数器
        init_lr = ops.init_lr # 学习率

        epochs_loss_dict = {}

        for epoch in range(0, ops.epochs):
            if ops.log_flag:
                sys.stdout = f_log
            print('\nepoch %d ------>>>'%epoch)
            model_.train()
            if loss_mean != 0.:
                if best_loss > (loss_mean/loss_idx):
                    flag_change_lr_cnt = 0
                    best_loss = (loss_mean/loss_idx)
                else:
                    flag_change_lr_cnt += 1

                    if flag_change_lr_cnt > 50:
                        init_lr = init_lr*ops.lr_decay
                        set_learning_rate(optimizer, init_lr)
                        flag_change_lr_cnt = 0


            loss_mean = 0.  # 损失均值
            loss_idx = 0.  # 损失计算计数器
            mse_list = []
            for i, (imgs_, pts_) in enumerate(dataloader):
                # print('imgs_, pts_',imgs_.size(), pts_.size())
                if use_cuda:
                    imgs_ = imgs_.cuda()  # pytorch 的 数据输入格式 ： (batch, channel, height, width)
                    pts_ = pts_.cuda()  # shape [batch_size,42]

                output = model_(imgs_.float())  # output 是21个点位(42个坐标)  shape [batch_size, 42]
                mse = calc_mes(output.cpu().detach().numpy(), pts_.cpu().float().numpy())  # 计算当前batch_size的mse
                # 将MSE指标写入TensorBoard日志
                writer.add_scalar('MSE', mse, global_step=epoch * len(dataloader) + i)

                mse_list.append(mse)
                
                if ops.loss_define == 'wing_loss':
                    loss = got_total_wing_loss(output, pts_.float())
                else:
                    loss = criterion(output, pts_.float())
                loss_mean += loss.item()
                loss_idx += 1.
                writer.add_scalar('mean_loss', loss_mean/loss_idx, global_step=epoch * len(dataloader) + i)

                loc_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print('  %s - %s - epoch [%s/%s] (%s/%s):' % (loc_time,ops.model,epoch,ops.epochs,i,int(dataset.__len__()/ops.batch_size)),\
                'Mean Loss : %.6f - Loss: %.6f'%(loss_mean/loss_idx,loss.item()),\
                ' lr : %.8f'%init_lr, ' bs :',ops.batch_size,\
                ' img_size: %s x %s'%(ops.img_size[0],ops.img_size[1]))
                # 计算梯度
                loss.backward()
                # 优化器对模型参数更新
                optimizer.step()
                # 优化器梯度清零
                optimizer.zero_grad()
                step += 1
            # 所有batch的mse求平均
            avg_mes = np.mean(mse_list)
            print("{}_avg_mes {}".format(epoch, avg_mes))
            torch.save(model_.state_dict(), ops.model_exp + '{}-size-{}-model_epoch-{}-avg_mse-{}.pth'.format(ops.model, ops.img_size[0], epoch, avg_mes))
        writer.close()
    except Exception as e:
        print('Exception : ',e) # 打印异常
        print('Exception  file : ', e.__traceback__.tb_frame.f_globals['__file__'])# 发生异常所在的文件
        print('Exception  line : ', e.__traceback__.tb_lineno)# 发生异常所在的行数

if __name__ == "__main__":


    
    