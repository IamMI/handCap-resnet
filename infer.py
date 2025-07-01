"""
Load pretrained params and take a inference to test
"""

import os
import argparse

from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
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
import matplotlib.pyplot as plt


def calc_mes(predicted, target):
    squared_diff = np.square(predicted - target)
    mes = np.mean(squared_diff)
    return mes

def infer(ops):
    model_ = resnet50(pretrained = False, num_classes = ops.num_classes,img_size = ops.img_size[0],dropout_factor=ops.dropout)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Dataset
    print("Loading Dataset....")
    dataset = LoadImagesAndLabels(ops=ops, img_size=ops.img_size,flag_agu=ops.flag_agu,fix_res = ops.fix_res,vis = False)
    print("handpose done")

    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    generator = torch.Generator().manual_seed(ops.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    print('len train datasets : %s, len eval datasets : %s'%(train_dataset.__len__(), val_dataset.__len__()))
    
    # Dataloader
    eval_dataloader = DataLoader(val_dataset,
                        batch_size=1,
                        num_workers=ops.num_workers,
                        shuffle=True,
                        pin_memory=False,
                        drop_last = True)
    
    # Load pretrained params
    if os.access(ops.fintune_model, os.F_OK):
        chkpt = torch.load(ops.fintune_model, map_location='cpu')
        print(model_.load_state_dict(chkpt))
        print('load fintune model : {}'.format(ops.fintune_model))
        del chkpt
    model_ = model_.to(device)
    
    # Inference
    model_.eval()
    imgs_list, output_list = [], []
    with torch.no_grad():
        for idx, (imgs_, pts_) in tqdm(enumerate(eval_dataloader)):
            imgs_ = imgs_.cuda()  
            pts_ = pts_.cuda()  
            output = model_(imgs_.float()) 
            
            imgs_list.append(imgs_.cpu())
            output_list.append(output.cpu())
            
            if idx >= 4:
                break
    
    hand_edges = [
        [0,1], [1,2], [2,3], [3,4],
        [0,5], [5,6], [6,7], [7,8],
        [0,9], [9,10], [10,11], [11,12],
        [0,13], [13,14], [14,15], [15,16],
        [0,17], [17,18], [18,19], [19,20]
    ]
    
    # Draw 
    for idx, (img, output) in enumerate(zip(imgs_list, output_list)):
        img = img.squeeze(0).numpy()  # [3, H, W]
        img = np.transpose(img, (1, 2, 0))  # [H, W, 3]
        img = (img * 256 + 128).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        h, w = img.shape[:2]
        pts = output.squeeze(0).numpy().reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h

        for x, y in pts:
            cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)

        for start, end in hand_edges:
            pt1 = (int(pts[start][0]), int(pts[start][1]))
            pt2 = (int(pts[end][0]), int(pts[end][1]))
            cv2.line(img, pt1, pt2, (255, 0, 0), 1)
        
        cv2.imwrite(f"results/hand_pred_{idx}.jpg", img)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=' Project Hand Train')
    parser.add_argument('--seed', type=int, default=126673, help='seed')  # 设置随机种子
    parser.add_argument('--model_exp', type=str, default='./model_exp', help='model_exp')  # 模型输出文件夹
    parser.add_argument('--model', type=str, default='resnet_50', help = 'model : resnet_34,resnet_50,resnet_101,squeezenet1_0,squeezenet1_1,shufflenetv2,shufflenet,mobilenetv2')  # 模型类型
    parser.add_argument('--num_classes', type=int, default=42, help = 'num_classes') #  landmarks 个数*2
    parser.add_argument('--GPUS', type=str, default='1', help='GPUS') # GPU选择
    parser.add_argument('--train_path', type=str,default = "F:/BaiduNetdiskDownload/handpose_datasets_v1-2021-01-31/handpose_datasets_v1/",help = 'datasets path')  # 训练集标注信息
    parser.add_argument('--pretrained', type=bool, default=False, help = 'imageNet_Pretrain')
    parser.add_argument('--fintune_model', type=str, default='resnet_50-size-256-loss-0.0642.pth', help = 'fintune_model')  # fintune model
    parser.add_argument('--loss_define', type=str, default = 'mse', help = 'define_loss wing_loss or mse')  # 损失函数定义
    parser.add_argument('--init_lr', type=float, default = 1e-3, help = 'init learning Rate')  # 初始化学习率
    parser.add_argument('--lr_decay', type=float, default=0.1, help = 'learningRate_decay')  # 学习率权重衰减率
    parser.add_argument('--weight_decay', type=float, default = 1e-6, help = 'weight_decay')  # 优化器正则损失权重
    parser.add_argument('--momentum', type=float, default = 0.9, help = 'momentum')  # 优化器动量
    parser.add_argument('--batch_size', type=int, default = 4,  help = 'batch_size')  # 训练每批次图像数量
    parser.add_argument('--dropout', type=float, default = 0.5, help = 'dropout')  # dropout
    parser.add_argument('--epochs', type=int, default=100,help = 'epochs')  # 训练周期
    parser.add_argument('--num_workers', type=int, default = 4,help = 'num_workers')  # 训练数据生成器线程数
    parser.add_argument('--img_size', type=tuple , default = (256,256),help = 'img_size')  # 输入模型图片尺寸
    parser.add_argument('--flag_agu', type=bool, default=False,help = 'data_augmentation')  # 训练数据生成器是否进行数据扩增
    parser.add_argument('--fix_res', type=bool, default=False,help = 'fix_resolution')  # 输入模型样本图片是否保证图像分辨率的长宽比
    parser.add_argument('--clear_model_exp', type=bool, default = False,help = 'clear_model_exp')  # 模型输出文件夹是否进行清除
    parser.add_argument('--log_flag', type=bool, default=False, help='log flag')  # 是否保存训练 log
    args = parser.parse_args()  


    infer(ops=args)
    
    