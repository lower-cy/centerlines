import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
import pywt

# ====================== 左图参数配置（灰度激光优化） ======================
LEFT_CONFIG = {
    # 基础参数
    'laser_color': 'gray',
    'min_laser_intensity': 100,  # 降低强度阈值适应暗区
    
    # 光照补偿
    'retinex_scale': 100,        # Retinex尺度参数(100-300)
    'illum_smooth': 45,          # 光照平滑核大小(奇数值)
    
    # 去噪参数
    'nlm_h': 15,                 # 非局部均值去噪强度(3-15)
    'wavelet_level': 4,          # 小波去噪层级(1-4)
    
    # 动态增强
    'adaptive_enhance': True,    # 启用自适应区域增强
    'enhance_power': 1.8,        # 局部增强强度(1.0-2.5)
    
    # 形态学参数
    'morph_kernel': (5, 11),     # 竖向结构元素尺寸
    'morph_iterations': 3,       # 形态学操作次数
    
    # 质心检测
    'dynamic_thresh_ratio': 0.3, # 动态阈值比例(0.2-0.6)
    'min_line_width': 4,         # 最小线宽(像素)
    'max_line_gap': 50,          # 最大允许断裂间距
    
    # 几何约束
    'roi_padding': 10,           # 边缘裁剪宽度
    'cluster_eps': 12,           # 聚类搜索半径
    'min_samples': 6,            # 最小聚类点数
    'min_line_length': 60,       # 有效线段最小长度
    
    # 后处理
    'smooth_sigma': 1.8,         # 高斯平滑强度
    'max_end_curvature': 0.15,   # 端点最大曲率
    
    # 新增参数
    'wavelet_type': 'bior1.3',   # 小波基类型
    'clahe_clip': 3.0            # CLAHE对比度限制
}

# ====================== 右图参数配置（红色激光优化） ======================
RIGHT_CONFIG = {
    'laser_color': 'red',
    'min_laser_intensity': 40,
    
    # 光照补偿
    'retinex_scale': 100,
    'illum_smooth': 35,
    
    # 去噪参数
    'nlm_h': 150,
    'wavelet_level': 2,
    
    # 动态增强
    'adaptive_enhance': True,
    'enhance_power': 2.0,
    
    # 形态学参数 
    'morph_kernel': (5, 15),
    'morph_iterations': 4,
    
    # 质心检测
    'dynamic_thresh_ratio': 0.35,
    'min_line_width': 3,
    'max_line_gap': 20,
    
    # 几何约束
    'roi_padding': 10,
    'cluster_eps': 15,
    'min_samples': 5,
    'min_line_length': 50,
    
    # 后处理
    'smooth_sigma': 1.2,
    'max_end_curvature': 0.2,
    
    # 新增参数
    'wavelet_type': 'bior1.3',
    'clahe_clip': 2.5
}

def wavelet_denoise(img, config):
    """
    小波阈值去噪
    Args:
        img: 输入灰度图像
        config: 配置参数
    Returns:
        去噪后的图像
    """
    coeffs = pywt.wavedec2(img, config['wavelet_type'], level=config['wavelet_level'])
    threshold = np.std(coeffs[-config['wavelet_level']]) * 0.7
    
    new_coeffs = []
    for c in coeffs:
        if isinstance(c, tuple):
            new_subbands = [pywt.threshold(sub, threshold, mode='soft') for sub in c]
            new_coeffs.append(tuple(new_subbands))
        else:
            new_c = pywt.threshold(c, threshold, mode='soft')
            new_coeffs.append(new_c)
    
    denoised = pywt.waverec2(new_coeffs, config['wavelet_type'])
    return np.clip(denoised, 0, 255).astype('uint8')

def retinex_illumination(img, config):
    """
    Retinex光照补偿
    Args:
        img: 输入BGR图像
        config: 配置参数
    Returns:
        光照校正后的灰度图像
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    illum = cv2.GaussianBlur(gray, (config['illum_smooth'],)*2, 0)
    illum = np.clip(illum, 1, 255)  # 防止除零
    
    normalized = (gray / illum) * config['retinex_scale']
    return cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def adaptive_illumination_compensation(img, config):
    """
    自适应光照补偿流程
    1. Retinex全局补偿
    2. CLAHE局部增强
    3. 非线性对比度调整
    """
    # 第一阶段：全局Retinex补偿
    global_comp = retinex_illumination(img, config)
    
    # 第二阶段：局部CLAHE增强
    clahe = cv2.createCLAHE(clipLimit=config['clahe_clip'], tileGridSize=(8,8))
    local_comp = clahe.apply(global_comp)
    
    # 第三阶段：非线性增强
    enhanced = np.power(local_comp/255.0, 1/config['enhance_power']) * 255
    return enhanced.astype(np.uint8)

def multi_scale_denoise(img, config):
    """
    多尺度去噪流程：
    1. 非局部均值去噪
    2. 小波阈值去噪
    3. 保边双边滤波
    """
    # 非局部均值去噪
    nlm = cv2.fastNlMeansDenoising(img, h=config['nlm_h'], 
                                 templateWindowSize=7, 
                                 searchWindowSize=21)
    
    # 小波去噪
    wavelet_denoised = wavelet_denoise(nlm, config)
    
    # 保边滤波
    bilateral = cv2.bilateralFilter(wavelet_denoised, 5, 25, 25)
    return bilateral

def dynamic_region_enhancement(img, config):
    """基于分水岭的区域自适应增强"""
    # 确保输入为单通道灰度图
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 转换为8位无符号整型
    img = img.astype(np.uint8)
    
    # 创建标记矩阵（必须为32位整数）
    markers = np.zeros(img.shape, dtype=np.int32)
    markers[img < 30] = 1
    markers[img > 200] = 2
    
    # 转换为三通道图像供分水岭使用
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color_img, markers)
    
    # 创建掩模
    dark_mask = (markers == 1).astype(np.uint8)*255
    bright_mask = (markers == 2).astype(np.uint8)*255
    
    # 暗区增强（显式处理数据类型）
    enhanced = img.astype(np.float32)  # 转换为浮点型进行运算
    blur_component = cv2.GaussianBlur(
        (img.astype(np.float32) * dark_mask.astype(np.float32))/255.0, 
        (0,0), 5
    )
    enhanced = cv2.addWeighted(
        enhanced, 1.0,
        blur_component, 0.6,
        0, dtype=cv2.CV_32F  # 显式指定输出类型
    )
    
    # 亮区抑制（确保数据类型一致）
    enhanced = enhanced.astype(np.uint8)  # 转换回uint8进行形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    bright_region = enhanced.copy()
    bright_region[bright_mask == 0] = 0  # 屏蔽非高光区
    bright_eroded = cv2.erode(bright_region, kernel, iterations=1)
    enhanced = cv2.addWeighted(enhanced, 1.0, bright_eroded, -0.3, 0)  # 轻度抑制
    
    return enhanced.astype(np.uint8)  # 确保最终输出为uint8

def enhance_laser_channel(img, config):
    """
    增强激光颜色通道
    Args:
        img: 输入图像（单通道或三通道）
        config: 配置参数
    Returns:
        增强后的三通道图像
    """
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    b, g, r = cv2.split(img)

    if config['laser_color'] == 'red':
        enhanced = cv2.addWeighted(r, 2.5, cv2.add(b, g), -1.0, 0)
    elif config['laser_color'] == 'green':
        enhanced = cv2.addWeighted(g, 2.5, cv2.add(r, b), -1.0, 0)
    else:  # blue
        enhanced = cv2.addWeighted(b, 2.5, cv2.add(r, g), -1.0, 0)

    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.merge([enhanced]*3)

def laser_preprocessing(img, config):
    """
    完整的激光图像预处理流水线：
    1. 光照补偿
    2. 多尺度去噪
    3. 区域增强
    4. 颜色通道增强
    """
    # 光照补偿
    illum_comp = adaptive_illumination_compensation(img, config)
    
    # 多尺度去噪
    denoised = multi_scale_denoise(illum_comp, config)
    
    # 动态区域增强
    if config['adaptive_enhance']:
        denoised = dynamic_region_enhancement(denoised, config)
    
    # 转换到BGR格式进行颜色增强
    enhanced = enhance_laser_channel(denoised, config)
    
    # 最终灰度化与CLAHE
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=config['clahe_clip'], tileGridSize=(8,8))
    return clahe.apply(gray)

def dynamic_centroid_detection(row, config):
    """
    改进的动态质心检测：
    - 自适应形态学闭合
    - 线宽动态判断
    - 加权质心计算
    """
    max_val = np.max(row)
    if max_val < config['min_laser_intensity']:
        return []

    thresh = max_val * config['dynamic_thresh_ratio']
    binary = np.where(row > thresh, 255, 0).astype(np.uint8)

    # 动态调整闭合核大小
    line_width = np.count_nonzero(binary) / (binary.shape[0]/255)
    kernel_size = max(3, int(line_width*0.7))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 线段检测
    segments = []
    start = -1
    for i, val in enumerate(closed):
        if val == 255 and start == -1:
            start = i
        elif val == 0 and start != -1:
            if i - start >= config['min_line_width']:
                segments.append((start, i-1))
            start = -1
    if start != -1 and len(closed)-start >= config['min_line_width']:
        segments.append((start, len(closed)-1))

    # 加权质心计算
    centers = []
    for s, e in segments:
        x = np.arange(s, e+1)
        weights = row[s:e+1]
        weights = (weights - np.min(weights)) ** 2  # 增强高亮区域权重
        if np.sum(weights) == 0:
            continue
        centroid = np.sum(x * weights) / np.sum(weights)
        centers.append(int(round(centroid)))

    return centers

def geometry_based_clustering(points, img_size, config):
    """
    改进的几何聚类：
    - 自适应ROI裁剪
    - 基于方向的DBSCAN聚类
    - 样条曲线拟合优化
    """
    h, w = img_size
    
    # 动态ROI裁剪
    x_std = np.std(points[:,0])
    roi_padding = min(config['roi_padding'], int(w*0.1))
    mask = (points[:,0] > roi_padding) & (points[:,0] < w - roi_padding)
    filtered = points[mask]

    # 方向增强的DBSCAN
    direction = np.arctan2(np.gradient(filtered[:,1]), np.gradient(filtered[:,0]))
    features = np.column_stack((filtered, direction*10))  # 方向特征加权
    
    db = DBSCAN(eps=config['cluster_eps'], min_samples=config['min_samples']).fit(features)
    
    valid_lines = []
    for label in set(db.labels_):
        if label == -1:
            continue
        
        cluster = filtered[db.labels_ == label]
        if len(cluster) < config['min_line_length']:
            continue
        
        # 按y坐标排序
        sorted_cluster = cluster[cluster[:,1].argsort()]
        
        try:
            # 样条曲线拟合
            tck, u = splprep(sorted_cluster.T, s=len(sorted_cluster)*0.8)
            new_u = np.linspace(u.min(), u.max(), int(len(u)*2))
            new_points = np.column_stack(splev(new_u, tck))
        except:
            new_points = sorted_cluster
        
        # 高斯平滑
        new_points[:,0] = gaussian_filter1d(new_points[:,0], config['smooth_sigma'])
        new_points[:,1] = gaussian_filter1d(new_points[:,1], config['smooth_sigma'])
        
        # 端点曲率过滤
        filtered_line = filter_endpoints_curvature(new_points, config)
        
        if len(filtered_line) >= config['min_line_length']:
            valid_lines.append(filtered_line)
    
    return valid_lines

def filter_endpoints_curvature(line, config):
    """
    改进的端点曲率分析：
    - 前/后10%点的曲率分析
    - 动态裁剪异常弯曲部分
    """
    if len(line) < 20:
        return line
    
    # 分析前段
    head = line[:10]
    dx = np.gradient(head[:,0])
    dy = np.gradient(head[:,1])
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    curvature = np.abs(d2x*dy - dx*d2y) / (dx**2 + dy**2 + 1e-6)**1.5
    if np.mean(curvature) > config['max_end_curvature']:
        line = line[5:]
    
    # 分析后段
    tail = line[-10:]
    dx = np.gradient(tail[:,0])
    dy = np.gradient(tail[:,1])
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    curvature = np.abs(d2x*dy - dx*d2y) / (dx**2 + dy**2 + 1e-6)**1.5
    if np.mean(curvature) > config['max_end_curvature']:
        line = line[:-5]
    
    return line

def detect_laser_lines(img, config):
    """
    激光线检测主流程
    1. 图像预处理
    2. 形态学优化
    3. 逐行质心检测
    4. 几何聚类
    """
    # 预处理
    preprocessed = laser_preprocessing(img, config)
    
    # 形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config['morph_kernel'])
    morph = cv2.morphologyEx(preprocessed, cv2.MORPH_CLOSE, kernel, 
                           iterations=config['morph_iterations'])
    
    # 质心检测
    points = []
    for y in range(morph.shape[0]):
        row = morph[y,:]
        centers = dynamic_centroid_detection(row, config)
        points.extend([[x, y] for x in centers])
    
    if len(points) < 100:  # 最少点数阈值
        return []
    
    # 几何聚类
    return geometry_based_clustering(np.array(points), morph.shape, config)

def visualize_results(img, lines, title):
    """增强可视化效果"""
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始图像
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title(f'{title} Original')
    
    # 预处理效果
    preprocessed = laser_preprocessing(img, LEFT_CONFIG if 'L' in title else RIGHT_CONFIG)
    ax[1].imshow(preprocessed, cmap='gray')
    ax[1].set_title('Preprocessed')
    
    # 检测结果
    vis = img.copy()
    colors = [(0,255,0), (255,0,0), (0,0,255)]
    for i, line in enumerate(lines):
        color = colors[i % len(colors)]
        pts = line.astype(int)
        cv2.polylines(vis, [pts], False, color, 2)
        if len(pts) > 0:
            cv2.circle(vis, tuple(pts[0]), 5, (0,0,255), -1)
            cv2.circle(vis, tuple(pts[-1]), 5, (255,0,0), -1)
    ax[2].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax[2].set_title(f'Detected {len(lines)} Lines')
    
    plt.tight_layout()
    plt.show()

def save_clean_lines(lines, filename):
    """保存清理后的激光线坐标"""
    with open(filename, 'w') as f:
        for i, line in enumerate(lines):
            clean_line = line[~np.isnan(line).any(axis=1)]
            np.savetxt(f, clean_line, fmt='%.2f',
                      header=f'Line {i+1}', comments='# ',
                      delimiter=',')

if __name__ == "__main__":
    # 加载测试图像
    left_img = cv2.imread('L.bmp')
    right_img = cv2.imread('R.jpg')
    
    print("处理左图(L.bmp)...")
    left_lines = detect_laser_lines(left_img, LEFT_CONFIG)
    print(f"检测到 {len(left_lines)} 条有效激光线")
    
    print("\n处理右图(R.jpg)...")
    right_lines = detect_laser_lines(right_img, RIGHT_CONFIG)
    print(f"检测到 {len(right_lines)} 条有效激光线")
    
    # 可视化与保存
    visualize_results(left_img, left_lines, 'Left Image')
    visualize_results(right_img, right_lines, 'Right Image')
    
    save_clean_lines(left_lines, 'left_clean_lines.csv')
    save_clean_lines(right_lines, 'right_clean_lines.csv')
    print("结果已保存为CSV文件")