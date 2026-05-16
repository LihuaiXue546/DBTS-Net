from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import torch

from dataset import ImgData
from model import UNet, SGVP, UNetWithSteger

def dice_coefficient(pred, label, smooth=1e-6):
    """计算 Dice 系数"""
    pred = pred.view(-1)
    label = label.view(-1)
    intersection = (pred * label).sum()
    return (2. * intersection + smooth) / (pred.sum() + label.sum() + smooth)
def train(net, device, path, epochs=40, bSize=1, lr=0.00001):
    igmData = ImgData(path)
    train_loader = DataLoader(igmData, bSize, shuffle=True)
    # 优化算法
    optimizer = optim.RMSprop(net.parameters(),
            lr=lr, weight_decay=1e-8, momentum=0.9)

    criterion = nn.BCEWithLogitsLoss()      # 损失函数
    bestLoss = float('inf')                # 最佳loss，初始化为无穷大

    # 训练epochs次
    for epoch in range(epochs):
        net.train()     # 训练模式
        epoch_loss = 0.0
        epoch_dice = 0.0
        num_batches = 0
        for image, label in train_loader:
            # 在训练过程中
            # 先梯度清零(与net.zero_grad()效果一样)
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            pred = net(image)   # 使用网络参数，输出预测结果
            loss = criterion(pred, label)   # 计算损失
            epoch_loss += loss.item()

            # 计算 Dice 系数
            pred_sigmoid = torch.sigmoid(pred)
            pred_binary = (pred_sigmoid > 0.5).float()  # 应用阈值
            dice = dice_coefficient(pred_binary, label)
            epoch_dice += dice.item()
            # 保存loss最小的网络参数 保存模型
            if loss < bestLoss:
                bestLoss = loss
                torch.save(net.state_dict(), 'best_model.pth')

            loss.backward() # 更新参数
            optimizer.step()
            num_batches += 1
        avg_loss = epoch_loss / num_batches
        avg_dice = epoch_dice / num_batches
        print(epoch, 'Loss/train:', avg_loss, 'Dice/train:', avg_dice)


def train_with_steger(net, device, path, epochs=40, bSize=1, lr=0.00001):
    # 使用Steger特征图加载数据
    igmData = ImgData(path, use_steger=True)
    train_loader = DataLoader(igmData, bSize, shuffle=True)
    # 优化算法
    optimizer = optim.RMSprop(net.parameters(),
            lr=lr, weight_decay=1e-8, momentum=0.9)

    criterion = nn.BCEWithLogitsLoss()      # 损失函数
    bestLoss = float('inf')                # 最佳loss，初始化为无穷大

    # 训练epochs次
    for epoch in range(epochs):
        net.train()     # 训练模式
        epoch_loss = 0.0
        epoch_dice = 0.0
        num_batches = 0
        for image, label in train_loader:
            # 在训练过程中
            # 先梯度清零(与net.zero_grad()效果一样)
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            pred = net(image)   # 使用网络参数，输出预测结果
            loss = criterion(pred, label)   # 计算损失
            epoch_loss += loss.item()

            # 计算 Dice 系数
            pred_sigmoid = torch.sigmoid(pred)
            pred_binary = (pred_sigmoid > 0.5).float()  # 应用阈值
            dice = dice_coefficient(pred_binary, label)
            epoch_dice += dice.item()
            # 保存loss最小的网络参数 保存模型
            if loss < bestLoss:
                bestLoss = loss
                torch.save(net.state_dict(), 'best_model_steger.pth')

            loss.backward() # 更新参数
            optimizer.step()
            num_batches += 1
        avg_loss = epoch_loss / num_batches
        avg_dice = epoch_dice / num_batches
        print(epoch, 'Loss/train_with_steger:', avg_loss, 'Dice/train_with_steger:', avg_dice)

# 主函数
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 使用SGVP (Steger Guided Vascular Prior)模型
    net = SGVP(2, 1)  # 输入通道数改为2
    net.to(device=device)
    
    path = "./data2/"
    # 使用新的训练函数
    train_with_steger(net, device, path, epochs=50)  # 增加训练轮数以充分学习
