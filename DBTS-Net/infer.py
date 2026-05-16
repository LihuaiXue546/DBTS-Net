import os
import torch
from tools_seg import Miou
import cv2
import argparse
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from models.deeplabv3.deeplabv3_model import DeepLabV3
from models.DBTSNet.DBTSNet_attention import DBTSNet
import sys
import torchvision.transforms.functional as tf
from dataset.create_dataset import Mydataset, for_train_transform, test_transform
import pandas as pd
from dataset.create_dataset import Mydataset_test
from collections import OrderedDict
import SimpleITK as sitk

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--imgs_val_path', '-iv', type=str,
                    default='', help='imgs val data path.')
parser.add_argument('--labels_val_path', '-lv', type=str,
                    default='', help='labels val data path.')
parser.add_argument('--resize', default=480, type=int, help='resize shape')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--imgs_val_list', '-cv', type=str,
                    default='./preprocessed_data/csv/val.csv', help='Path to validation csv list')
parser.add_argument('--model_type', type=str, default='DBTS-Net',
                    choices=['DBTS-Net'],
                    help='Model type to use (default: DBTS-Net)')
parser.add_argument('--encoder_name', type=str, default='resnext101_32x4d',
                    help='Encoder name for the model')
parser.add_argument('--model_path', type=str, 
                    default='./checkpoint/T2_Deeplabv3_resnext101_32x4d/ckptlast.pth',
                    help='Path to the model checkpoint file')
parser.add_argument('--custom_save_dir', type=str, 
                    default='',
                    help='Custom directory containing model checkpoints (overrides model_path if specified)')
parser.add_argument('--use_best_model', action='store_true',
                    help='Use best model (ckpt.pth) instead of last model (ckptlast.pth) from custom_save_dir')
parser.add_argument('--result_dir', type=str, 
                    default='./preprocessed_data/result/',
                    help='Directory to save inference results')
parser.add_argument('--devicenum', type=str, default='0', 
                    help='CUDA device to use, e.g., "0" or "0,1"')
args = parser.parse_args()

# 设置CUDA设备
os.environ['CUDA_VISIBLE_DEVICES'] = args.devicenum
result_path = args.result_dir
if not os.path.exists(result_path):
    os.makedirs(result_path, exist_ok=True)
label_path = os.path.join(result_path, 'label')
if not os.path.exists(label_path):
    os.makedirs(label_path, exist_ok=True)
val_csv = pd.read_csv(args.imgs_val_list)#[:30]

val_imgs, val_masks = val_csv['image_name'], val_csv['image_name']

# 从CSV文件中获取数据路径，确保与训练时的路径结构一致
# 这里假设CSV中的image_name包含完整文件名，需要根据实际情况调整
val_imgs = [''.join(['./preprocessed_data/val/image','/',i]) for i in val_imgs]
val_masks = [''.join(['./preprocessed_data/val/label','/',i]) for i in val_masks]

train_transform = for_train_transform()
test_transform = test_transform
valset = Mydataset_test(val_imgs, val_masks, test_transform)
valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=4)
print('==> Preparing data..')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model
print('==> Building model..')

# 使用DBTS-Net模型
model = DBTSNet(encoder_name=args.encoder_name, encoder_weights=None, classes=2)


model = model.to(device)


# 确定模型路径
model_path = args.model_path
# 如果指定了自定义保存目录，则优先使用该目录下的模型文件
if args.custom_save_dir:
    if args.use_best_model:
        model_path = os.path.join(args.custom_save_dir, 'ckpt.pth')
    else:
        model_path = os.path.join(args.custom_save_dir, 'ckptlast.pth')
    print(f"使用自定义保存目录下的模型: {model_path}")

try:
    state_dict = torch.load(model_path)
    
    # 过滤掉不匹配的权重键（如果仍然有问题）
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
    
    # 更新模型状态字典
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)
    
    # 打印加载的权重信息
    print(f"成功加载模型权重，加载了 {len(filtered_state_dict)} / {len(state_dict)} 个权重参数")
except Exception as e:
    print(f"加载模型失败: {e}")
    sys.exit(1)
model = model.to(device)
model.eval()

import csv
from PIL import Image
import time

begin_time = time.time()

imwrite_image = True


def train_val():
    with torch.no_grad():
        PA = 0
        list = []
        number = 0
        train_dice = 0
        train_jaccard = 0  # 前景IoU
        train_recall = 0
        train_precision = 0
        train_hd95 = 0
        # 初始化样本名称和dice值列表
        sample_names = []
        sample_dices = []
        for batch_idx, (name, imgs, masks) in enumerate(valloader):
            number += 1
            # print(img_path)
            sys.stdout.write('\r%d/%s' % (number, len(valloader)))
            batch_idx += 1
            imgs, masks_cuda = imgs.to(device), masks.to(device)

            imgs = imgs.float()
            masks_pred = model(imgs)
            # masks_pred = masks_pred[0]

            predicted = masks_pred.argmax(1)
            # 启用预测结果和标签的二值化处理
            predicted[predicted > 0] = 1
            masks_cuda[masks_cuda > 0] = 1
            if torch.sum(masks_cuda) == 0 and torch.sum(predicted) == 0:
                number -= 1
                # train_mdice += 1
                # train_miou += 1
                # train_jaccard += 1
                # train_accuracy += 1
                # train_dice += 1
                # train_recall += 1
                # train_SP += 1
                # train_precision += 1
                # train_f1+= 1

            else:
                # 计算当前样本的dice值
                current_dice = Miou.dice(predicted, masks_cuda).item()
                
                # 添加到样本列表
                sample_names.append(name[0])
                sample_dices.append(current_dice)
                
                # 打印当前样本信息
                print(f"样本 {name[0]} 的Dice值: {current_dice:.4f}")
                
                # 计算前景IoU (使用jaccard函数，只计算前景)
                train_jaccard += Miou.jaccard(predicted, masks_cuda).item()
                train_dice += current_dice  # 使用已计算的dice值
                train_recall += Miou.recall(predicted, masks_cuda).item()
                train_precision += Miou.precision(predicted, masks_cuda).item()
                
                # 计算HD95
                pred_np = predicted.cpu().numpy().squeeze()
                target_np = masks_cuda.cpu().numpy().squeeze()
                # 转换为SimpleITK图像
                pred_sitk = sitk.GetImageFromArray(pred_np.astype(np.uint8))
                target_sitk = sitk.GetImageFromArray(target_np.astype(np.uint8))
                # 设置默认间距
                pred_sitk.SetSpacing([1.0, 1.0])
                target_sitk.SetSpacing([1.0, 1.0])
                # 计算HD95
                try:
                    hausdorff_filter = sitk.HausdorffDistanceImageFilter()
                    hausdorff_filter.SetUseImageSpacing(True)
                    hausdorff_filter.Execute(pred_sitk, target_sitk)
                    hd95 = hausdorff_filter.GetAverageHausdorffDistance()
                except:
                    hd95 = 0.0
                train_hd95 += hd95


            #  softmax
            if imwrite_image:
                predict = predicted.squeeze(0)
                mask_np = predict.cpu().numpy()  # np.array
                mask_np = (mask_np * 255).astype('uint8')
                mask_np[mask_np > 0] = 255
                cv2.imwrite(os.path.join(result_path, name[0]), mask_np)
                masks_cuda = masks_cuda.squeeze(0)
                label_np = masks_cuda.cpu().numpy()  # np.array
                label_np = (label_np * 255).astype('uint8')
                label_np[label_np > 0] = 255
                masks_cuda_max = torch.max(masks_cuda)
                label_np_max = np.max(label_np)
                cv2.imwrite(os.path.join(label_path, name[0]), label_np)

        end_time = time.time()
        print('\n')
        print("时间")
        print(end_time - begin_time)
        print('前景IoU')
        print(train_jaccard / number)
        print('dice')
        print(train_dice / number)
        print('recall')
        print(train_recall / number)
        print('precision')
        print(train_precision / number)
        print('HD95')
        print(train_hd95 / number)
        
        # 输出每个样本的dice值汇总
        print('\n每个样本的Dice值汇总:')
        for i, (sample_name, dice_value) in enumerate(zip(sample_names, sample_dices)):
            print(f"{i+1}. 样本: {sample_name}, Dice值: {dice_value:.4f}")
        
        # 计算并输出统计信息
        avg_dice = sum(sample_dices) / len(sample_dices) if sample_dices else 0
        print(f"\nDice值统计:")
        print(f"最小值: {min(sample_dices):.6f}")
        print(f"最大值: {max(sample_dices):.6f}")
        print(f"平均值: {avg_dice:.6f}")
        print(f"中位数: {sorted(sample_dices)[len(sample_dices) // 2]:.6f}")
        
        # 将结果保存到txt文件
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"evaluation_results_{timestamp}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("模型评估结果\n")
            f.write(f"评估时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("整体评估指标:\n")
            f.write(f"平均Dice系数: {train_dice / number:.6f}\n")
            f.write(f"平均前景IoU: {train_jaccard / number:.6f}\n")
            f.write(f"平均Recall: {train_recall / number:.6f}\n")
            f.write(f"平均Precision: {train_precision / number:.6f}\n")
            f.write(f"平均HD95: {train_hd95 / number:.6f}\n\n")
            
            f.write("各样本Dice值:\n")
            for name, dice in zip(sample_names, sample_dices):
                f.write(f"{name}: {dice:.6f}\n")
            
            f.write("\nDice值统计信息:\n")
            f.write(f"最小值: {min(sample_dices):.6f}\n")
            f.write(f"最大值: {max(sample_dices):.6f}\n")
            f.write(f"平均值: {avg_dice:.6f}\n")
            f.write(f"中位数: {sorted(sample_dices)[len(sample_dices) // 2]:.6f}\n")
        
        print(f"\n评估结果已保存到: {output_file}")

# 40s 单张按批次  226.8个合一batch  145 4角度
if __name__ == '__main__':
    train_val()

