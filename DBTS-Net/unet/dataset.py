from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch

class ImgData(Dataset):
    def __init__(self, data_path, use_steger=True):
        self.path = data_path
        self.imgForder = os.path.join(data_path, "image")
        self.use_steger = use_steger

    # 应用窗宽窗位调整图像对比度
    def apply_window_level(self, image, level=240, window=800):
        min_val = level - window/2
        max_val = level + window/2
        windowed = np.clip(image, min_val, max_val)
        windowed = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        return windowed
    
    # 使用Steger算法提取血管中心线特征
    def steger_feature_extraction(self, img):
        # 应用窗宽窗位
        img_windowed = self.apply_window_level(img)
        
        # 多尺度高斯滤波参数
        num = 5  # 减少尺度数量以提高效率
        Ks = [1, 2, 3, 4, 5]
        Sigmas = [1.0, 0.8, 0.6, 0.4, 0.2]
        
        # 多尺度高斯滤波
        float_src = img_windowed.astype(np.float32)
        gaussian_ims = []
        for i in range(num):
            blurred = cv2.GaussianBlur(float_src, (Ks[i]*2+1, Ks[i]*2+1), Sigmas[i])
            gaussian_ims.append(blurred)
        
        # 创建自适应尺度响应图像
        res = np.zeros_like(float_src)
        rows, cols = res.shape
        pts = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        for i in range(rows):
            for j in range(cols):
                if img_windowed[i, j] > 50:  # 阈值过滤
                    cnt2 = float('inf')
                    for dx, dy in pts:
                        cnt1 = 0
                        x, y = j + (cnt1+1)*dx, i + (cnt1+1)*dy
                        while 0 <= x < cols and 0 <= y < rows and img_windowed[y, x] > 50 and cnt1 < cnt2:
                            cnt1 += 1
                            x, y = j + (cnt1+1)*dx, i + (cnt1+1)*dy
                        if cnt1 < cnt2:
                            cnt2 = cnt1
                    
                    l = 0
                    while l < num and Ks[l] <= cnt2:
                        l += 1
                    if l < num:
                        res[i, j] = gaussian_ims[l][i, j]
        
        # 计算导数
        m1 = np.array([[1, -1]], dtype=np.float32)  # x一阶导
        m2 = np.array([[1], [-1]], dtype=np.float32)  # y一阶导
        dx = cv2.filter2D(res, -1, m1)
        dy = cv2.filter2D(res, -1, m2)
        
        # 计算梯度幅度作为特征图
        gradient_magnitude = np.sqrt(dx*dx + dy*dy)
        gradient_magnitude = (gradient_magnitude - gradient_magnitude.min()) / \
                           (gradient_magnitude.max() - gradient_magnitude.min() + 1e-8) * 255
        
        return gradient_magnitude.astype(np.uint8)
    
    # 加载图像
    def loadImg(self, path):
        img = np.array(Image.open(path))
        
        if self.use_steger:
            # 生成Steger特征图
            steger_feature = self.steger_feature_extraction(img)
            # 组合原始图像和Steger特征图作为双通道输入
            combined = np.stack((img, steger_feature), axis=0)
            return combined
        else:
            # 保持单通道输入
            return img.reshape(1, *img.shape)

    # 根据index读取图片
    def __getitem__(self, index):
        pImg = os.path.join(self.path, f"image/{index}.png")
        pLabel = os.path.join(self.path, f"label/{index}.png")
        
        # 加载图像
        img = np.array(Image.open(pImg))
        label = np.array(Image.open(pLabel))
        
        # 数据归一化
        img = img.astype(np.float32) / 255.0
        
        # 生成Steger特征图
        if self.use_steger:
            steger_feature = self.steger_feature_extraction((img * 255).astype(np.uint8))
            steger_feature = steger_feature.astype(np.float32) / 255.0
            # 组合原始图像和Steger特征图
            combined = np.stack((img, steger_feature), axis=0)
            image = combined
        else:
            image = img.reshape(1, *img.shape)
        
        # 标签处理
        label = label.reshape(1, *label.shape)
        if label.max() > 1:
            label = label / 255.0
        
        # 随机翻转图像，增加训练样本
        flipCode = np.random.randint(3)
        if flipCode != 0:
            image = np.flip(image, flipCode).copy()
            label = np.flip(label, flipCode).copy()
        
        # 转换为torch张量
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(os.listdir(self.imgForder))
