import torch
from PIL import Image
import numpy as np
import os
import cv2

from dataset import ImgData
from model import UNet, SGVP, UNetWithSteger

def apply_window_level(image, level=240, window=800):
    """应用窗宽窗位调整图像对比度"""
    min_val = level - window/2
    max_val = level + window/2
    windowed = np.clip(image, min_val, max_val)
    windowed = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return windowed

def steger_feature_extraction(img):
    """使用Steger算法提取血管中心线特征"""
    # 应用窗宽窗位
    img_windowed = apply_window_level(img)
    
    # 多尺度高斯滤波参数
    num = 5
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
    if gradient_magnitude.max() > gradient_magnitude.min():
        gradient_magnitude = (gradient_magnitude - gradient_magnitude.min()) / \
                           (gradient_magnitude.max() - gradient_magnitude.min()) * 255
    
    return gradient_magnitude.astype(np.uint8)

def predictOne(net, device, pRead, pSave, use_steger=True):
    """预测单个图像，支持使用Steger特征"""
    img = Image.open(pRead)
    img_np = np.array(img)
    
    if use_steger:
        # 数据归一化
        img_normalized = img_np.astype(np.float32) / 255.0
        
        # 生成Steger特征图
        steger_feature = steger_feature_extraction(img_np)
        steger_feature = steger_feature.astype(np.float32) / 255.0
        
        # 组合原始图像和Steger特征图
        combined = np.stack((img_normalized, steger_feature), axis=0)
        img_tensor = torch.from_numpy(combined).unsqueeze(0)
    else:
        # 单通道输入
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    
    with torch.no_grad():
        pred = net(img_tensor)     # 预测
    
    # 应用sigmoid并转换为二值图像
    pred_sigmoid = torch.sigmoid(pred)
    pred_binary = (pred_sigmoid >= 0.5).float() * 255
    
    pred = np.array(pred_binary.data.cpu()[0])[0]
    img = Image.fromarray(pred.astype(np.uint8))
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(pSave), exist_ok=True)
    img.save(pSave)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 使用SGVP (Steger Guided Vascular Prior)模型
    net = SGVP(2, 1)  # 输入通道数为2
    net.to(device=device)
    
    try:
        # 尝试加载集成Steger特征的模型权重
        net.load_state_dict(torch.load('best_model_steger.pth', map_location=device))
        print("使用集成Steger特征的模型进行预测")
        use_steger = True
    except FileNotFoundError:
        # 如果找不到集成Steger特征的模型，则使用原始模型
        print("找不到Steger模型权重，使用原始UNet模型")
        net = UNet(1, 1)
        net.to(device=device)
        net.load_state_dict(torch.load('best_model.pth', map_location=device))
        use_steger = False
    
    net.eval()  # 测试模式
    
    # 创建预测结果保存目录
    predict_dir = "./data2/predict_steger" if use_steger else "./data2/predict"
    os.makedirs(predict_dir, exist_ok=True)
    
    # 获取测试图像列表
    test_dir = './data2/test'
    fs = os.listdir(test_dir)
    
    # 预测所有测试图像
    for f in fs:
        pRead = os.path.join(test_dir, f)
        pSave = os.path.join(predict_dir, f)
        predictOne(net, device, pRead, pSave, use_steger)
        print(f"预测完成: {f}")

if __name__ == "__main__":
    main()
