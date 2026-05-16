import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt

def get_gray(im, col, row):
    """获取图像指定位置的灰度值"""
    return im[row, col]

def refine(image):
    """细化算法，去除中心线周围的冗余像素"""
    p = [0] * 8
    del_points = []
    height, width = image.shape[:2]

    while True:
        del_points.clear()

        # 上下收缩处理
        for i in range(1, height-1):
            for j in range(1, width-1):
                grayvalue = get_gray(image, j, i)
                if grayvalue != 0:
                    p[0] = 0 if get_gray(image, j+1, i) == 0 else 1
                    p[1] = 0 if get_gray(image, j+1, i-1) == 0 else 1
                    p[2] = 0 if get_gray(image, j, i-1) == 0 else 1
                    p[3] = 0 if get_gray(image, j-1, i-1) == 0 else 1
                    p[4] = 0 if get_gray(image, j-1, i) == 0 else 1
                    p[5] = 0 if get_gray(image, j-1, i+1) == 0 else 1
                    p[6] = 0 if get_gray(image, j, i+1) == 0 else 1
                    p[7] = 0 if get_gray(image, j+1, i+1) == 0 else 1

                    if i < height-2:
                        down = 0 if get_gray(image, j, i+2) == 0 else 1
                    else:
                        down = 1

                    if p[6] and (p[5] or p[7] or p[0] or p[4]) and not (p[1] or p[3]) and p[2] == 0 and down:
                        del_points.append((j, i))
                    if p[2] and (p[1] or p[3] or p[0] or p[4]) and not (p[5] or p[7]) and p[6] == 0:
                        del_points.append((j, i))

        # 左右收缩处理
        for i in range(1, height-1):
            for j in range(1, width-1):
                grayvalue = get_gray(image, j, i)
                if grayvalue != 0:
                    p[0] = 0 if get_gray(image, j+1, i) == 0 else 1
                    p[1] = 0 if get_gray(image, j+1, i-1) == 0 else 1
                    p[2] = 0 if get_gray(image, j, i-1) == 0 else 1
                    p[3] = 0 if get_gray(image, j-1, i-1) == 0 else 1
                    p[4] = 0 if get_gray(image, j-1, i) == 0 else 1
                    p[5] = 0 if get_gray(image, j-1, i+1) == 0 else 1
                    p[6] = 0 if get_gray(image, j, i+1) == 0 else 1
                    p[7] = 0 if get_gray(image, j+1, i+1) == 0 else 1

                    if j < width-2:
                        right = 0 if get_gray(image, j+2, i) == 0 else 1
                    else:
                        right = 1

                    if p[0] and (p[1] or p[7] or p[2] or p[6]) and not (p[3] or p[5]) and p[4] == 0 and right:
                        del_points.append((j, i))
                    if p[4] and (p[3] or p[5] or p[2] or p[6]) and not (p[1] or p[7]) and p[0] == 0:
                        del_points.append((j, i))

        if not del_points:
            break

        for pt in del_points:
            image[pt[1], pt[0]] = 0

def apply_window_level(image, level=127.5, window=255.0):
    """应用窗宽窗位调整图像对比度"""
    min_val = max(0, level - window/2)  # 确保不低于0
    max_val = min(255, level + window/2)  # 确保不超过255
    windowed = np.clip(image, min_val, max_val)
    if (max_val - min_val) > 0:
        windowed = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return windowed

def compute_eigen(hessian):
    """计算Hessian矩阵的特征值和特征向量"""
    try:
        eigenvalues, eigenvectors = np.linalg.eig(hessian)
        return eigenvalues, eigenvectors
    except:
        return None, None

def debug_filter_results(gaussian_ims, src0, output_dir=None):
    """调试和显示滤波结果"""
    print("调试滤波结果...")

    plt.figure(figsize=(15, 10))

    # 显示原始图像
    plt.subplot(3, 4, 1)
    plt.imshow(src0, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')

    # 显示几个关键尺度的滤波结果
    scales_to_show = [0, 3, 6, 9]  # 显示第1,4,7,10个尺度
    for idx, scale_idx in enumerate(scales_to_show):
        if scale_idx < len(gaussian_ims):
            plt.subplot(3, 4, idx + 2)
            filtered_normalized = cv2.normalize(gaussian_ims[scale_idx], None, 0, 255, cv2.NORM_MINMAX)
            plt.imshow(filtered_normalized, cmap='gray')
            plt.title(f'尺度 {scale_idx + 1}')
            plt.axis('off')

            # 打印统计信息
            print(f"尺度 {scale_idx + 1}: min={np.min(gaussian_ims[scale_idx]):.2f}, "
                  f"max={np.max(gaussian_ims[scale_idx]):.2f}, "
                  f"mean={np.mean(gaussian_ims[scale_idx]):.2f}")

    # 显示所有尺度的均值结果
    combined_result = np.mean(gaussian_ims, axis=0)
    plt.subplot(3, 4, 6)
    combined_normalized = cv2.normalize(combined_result, None, 0, 255, cv2.NORM_MINMAX)
    plt.imshow(combined_normalized, cmap='gray')
    plt.title('多尺度融合结果')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 保存调试结果
    if output_dir:
        for i, filtered in enumerate(gaussian_ims):
            filtered_normalized = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, f"debug_scale_{i+1}.png"), filtered_normalized)

def simplified_steger_centerline_detection(image_path, output_dir=None):
    """简化的Steger中心线提取方法，避免滤波问题"""

    # 记录开始时间
    start_time = time.time()

    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：文件不存在 - {image_path}")
        return None, None

    # 1. 加载图像
    print("加载图像...")
    src0 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if src0 is None:
        print(f"错误：无法读取图像 - {image_path}")
        return None, None

    print(f"图像尺寸: {src0.shape}")

    # 创建输出目录
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. 简化的预处理
    print("预处理...")

    # 方法1: 直接使用原图或轻微增强
    # 使用直方图均衡化增强对比度
    src0_enhanced = cv2.equalizeHist(src0)

    # 方法2: 或者使用自适应直方图均衡化（更适合血管图像）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    src0_enhanced = clahe.apply(src0)

    color_src0 = cv2.cvtColor(src0_enhanced, cv2.COLOR_GRAY2BGR)

    # 保存预处理结果
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, "1_preprocessed.png"), src0_enhanced)

    # 3. 简化的滤波方法 - 使用单一尺度高斯滤波
    print("高斯滤波...")

    # 选择合适的sigma值
    sigma = 1.5  # 适中的平滑程度
    ksize = 5    # 核大小

    # 应用高斯滤波
    float_src = src0_enhanced.astype(np.float32)
    filtered = cv2.GaussianBlur(float_src, (ksize, ksize), sigma)

    print(f"滤波后范围: [{np.min(filtered):.2f}, {np.max(filtered):.2f}]")

    # 保存滤波结果
    if output_dir:
        filtered_normalized = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, "2_filtered.png"), filtered_normalized)

    # 4. 计算导数（使用滤波后的图像）
    print("计算导数...")

    # 使用Sobel算子计算导数
    dx = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
    dxx = cv2.Sobel(dx, cv2.CV_32F, 1, 0, ksize=3)
    dyy = cv2.Sobel(dy, cv2.CV_32F, 0, 1, ksize=3)
    dxy = cv2.Sobel(dx, cv2.CV_32F, 0, 1, ksize=3)

    # 5. Hessian矩阵中心线检测
    print("Hessian矩阵中心线检测...")

    rows, cols = src0_enhanced.shape
    centerline = np.zeros_like(src0_enhanced, dtype=np.uint8)

    # 创建二值掩码来确定感兴趣区域
    _, binary_mask = cv2.threshold(src0_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    centerline_points = 0
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if binary_mask[i, j] > 128:  # 只在前景区域检测
                # 构建Hessian矩阵
                hessian = np.array([
                    [dxx[i, j], dxy[i, j]],
                    [dxy[i, j], dyy[i, j]]
                ], dtype=np.float32)

                # 计算特征值和特征向量
                eigenvalues, eigenvectors = compute_eigen(hessian)

                if eigenvalues is not None and eigenvectors is not None and len(eigenvalues) >= 2:
                    # 选择绝对值较大的特征值
                    idx = np.argmax(np.abs(eigenvalues))
                    nx, ny = eigenvectors[idx, 0], eigenvectors[idx, 1]

                    # 计算中心线条件
                    denominator = nx*nx*dxx[i, j] + 2*nx*ny*dxy[i, j] + ny*ny*dyy[i, j]
                    if abs(denominator) > 1e-10:
                        t = -(nx*dx[i, j] + ny*dy[i, j]) / denominator

                        # 中心线条件
                        if abs(t*nx) <= 0.8 and abs(t*ny) <= 0.8:
                            centerline[i, j] = 255
                            centerline_points += 1

    print(f"检测到 {centerline_points} 个中心线点")

    # 保存原始中心线
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, "3_centerline_raw.png"), centerline)

    # 6. 中心线后处理
    print("中心线后处理...")

    # 去除小噪声
    kernel = np.ones((3,3), np.uint8)
    centerline_cleaned = cv2.morphologyEx(centerline, cv2.MORPH_OPEN, kernel)

    # 细化中心线
    refine(centerline_cleaned)

    # 保存细化后的中心线
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, "4_centerline_refined.png"), centerline_cleaned)

    # 7. 在原图上标记中心线
    result_image = color_src0.copy()
    result_image[centerline_cleaned == 255] = [0, 0, 255]

    # 保存最终结果
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, "5_final_result.png"), result_image)

    # 计算处理时间
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"处理完成! 总耗时: {processing_time:.2f}秒")

    # 8. 显示结果
    print("显示结果...")

    # 调整显示尺寸
    max_display_size = 800
    height, width = src0.shape
    scale = min(max_display_size/width, max_display_size/height)

    if scale < 1:
        display_size = (int(width*scale), int(height*scale))
        src0_display = cv2.resize(src0_enhanced, display_size)
        centerline_display = cv2.resize(centerline_cleaned, display_size)
        result_display = cv2.resize(result_image, display_size)
    else:
        src0_display = src0_enhanced
        centerline_display = centerline_cleaned
        result_display = result_image

    cv2.imshow("1. 预处理图像", src0_display)
    cv2.imshow("2. 滤波结果", cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    cv2.imshow("3. 提取的中心线", centerline_display)
    cv2.imshow("4. 最终结果", result_display)

    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return centerline_cleaned, result_image

# 主函数
if __name__ == "__main__":
    # 设置图像路径
    image_path = r""
    output_dir = r""

    # 运行简化的中心线提取
    centerline, result = simplified_steger_centerline_detection(image_path, output_dir)

    # 打印结果信息
    if centerline is not None:
        centerline_pixels = np.sum(centerline == 255)
        print(f"提取的中心线包含 {centerline_pixels} 个像素点")
        print(f"结果已保存到: {output_dir}")
    else:
        print("中心线提取失败")