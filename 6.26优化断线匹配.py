import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, SpectralClustering
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
import matplotlib.cm as cm

# ====================== 相机参数配置 ======================
CAMERA_MATRIX_LEFT = np.array([
    [4703.3840666469305, 0.0, 1133.8966264844476],
    [0.0, 4657.770006641158, 983.7755276735744],
    [0.0, 0.0, 1.0]
])

CAMERA_MATRIX_RIGHT = np.array([
    [4409.199175099535, 0.0, 1531.0013908252736],
    [0.0, 4384.905205883512, 1013.4751888939345],
    [0.0, 0.0, 1.0]
])

DIST_COEFF_LEFT = np.array([-0.19060368249367288, -6.827044122904246, 0.015377030028687984, -0.00750634791176898, 107.39588017569562])
DIST_COEFF_RIGHT = np.array([-0.42270673798875497, 1.378263372731151, 0.009909410979026863, -0.008593483642757997, -1.0961258361436514])

R = np.array([
    [0.9867230542685737, 0.007483211056180142, 0.1622393778562597],
    [-0.005753664364150946, 0.9999215317777955, -0.011127696685821956],
    [-0.16230991812357692, 0.010046483933974946, 0.9866886837494805]
])

T = np.array([-65.930698300496, 0.7317230319931822, -12.020455702540955])

# ====================== 左图参数配置（针对灰度图优化） ======================
LEFT_CONFIG = {
    # 基础参数
    'laser_color': 'gray',       # 激光颜色类型
    'min_laser_intensity': 75,  # [30-100] 最低有效激光强度
    
    # 预处理参数
    'clahe_clip': 3.5,          # [1.0-5.0] 对比度增强上限
    'blur_kernel': (3, 3),      # 高斯模糊核大小
    'gamma_correct': 1.0,       # 伽马校正系数
    'specular_thresh': 200,     # 高光检测阈值
    
    # 局部增强参数
    'local_enhance_region': (0, 1),  # 右侧1/3区域增强
    'clahe_clip_local': 1.5,
    'blend_weights': (0.2, 0.8),

    # 形态学参数
    'morph_kernel': (5, 11),    # 竖向特征检测
    'morph_iterations': 4,
    
    # 质心检测
    'dynamic_thresh_ratio':0.6, # 动态阈值比例
    'min_line_width': 1,        # 最小有效线宽
    'max_line_gap': 200,         # 断裂容忍度

    # 几何约束
    'roi_padding': 10,          # 边缘裁剪
    'cluster_eps': 6,          # 更小聚类半径（适应结构光连续性）
    'min_samples': 6,          # 最小样本极数
    'min_line_length': 80,      # 有效线段长度

    # 后处理
    'smooth_sigma': 2.5,        # 平滑强度
    'max_end_curvature': 0.08, # 更严格的端点曲率限制
    'smooth_degree': 3.0,       # 插值平滑度
    
    # 断线匹配参数
    'max_gap_for_matching': 500,  # 最大匹配间隙（像素）
    'direction_similarity': 0.2,  # 方向相似度阈值（cosθ）
    'intensity_similarity': 0.8,  # 强度相似度阈值
    'position_tolerance': 30,     # 位置容忍度（像素）
    'min_extension_length': 50,   # 最小延长线长度
    'max_extension_angle': 60,    # 最大延长角度(度),
    'breakpoint_threshold': 0.3,  # 断线匹配阈值
    
    # 立体匹配参数
    'min_depth': 200,           # 最小深度(mm)
    'max_depth': 600,          # 最大深度(mm)
    'disparity_tolerance': 5,   # 视差容差(像素)
    'width_diff_threshold': 0.3, # 宽度差异阈值
    'min_matched_lines': 3,     # 最小匹配线段数
}

# ====================极== 右图参数配置（针对彩色图优化） ====================== 
RIGHT_CONFIG = {
    # 基础参数
    'laser_color': 'red',        # 激光颜色类型
    'min_laser_intensity': 75,   # 最低有效激光强度

    # 预处理参数
    'clahe_clip': 2.0,           # 对比度增强上限
    'blur_kernel': (3, 3),       # 高斯模糊核大小
    'gamma_correct': 0.75,       # 高光抑制
    'specular_thresh': 180,      # 高光检测阈值

    # 局部增强参数
    'local_enhance_region': (0, 1),  # 右侧1/4区域增强
    'clahe_clip_local': 5.0,
    'blend_weights': (0.2, 0.8),

    # 形态学参数
    'morph_kernel': (5, 11),     # 竖向特征检测
    'morph_iterations': 4,

    # 质心检测
    'dynamic_thresh_ratio': 0.25,# 抗噪阈值
    'min_line_width': 1,         # 激光线宽度
    'max_line_gap': 200,          # 断裂容忍度

    # 几何约束
    'roi_padding': 15,           # 边缘裁剪
    'cluster_eps': 6,            # 更小聚类半径
    'min_samples': 6,           # 更小样本数
    'min_line_length': 100,      # 有效线段长度

    # 后处理
    'smooth_sigma': 2.0,         # 平滑强度
    'max_end_curvature': 0.15,   # 端点曲率限制
    'smooth_degree': 2.5,        # 插值平滑度
    
    # 断线匹配参数
    'max_gap_for_matching': 500,   # 最大匹配间隙
    'direction_similarity': 0.2, # 方向相似度阈值
    'intensity_similarity': 0.75, # 强度相似度阈值
    'position_tolerance': 20,     # 位置容忍度
    'min_extension_length': 40,   # 最小延长线长度
    'max_extension_angle': 60,    # 最大延长角度(度),
    'breakpoint_threshold': 0.3,  # 断线匹配阈值
    
    # 立体匹配参数
    'min_depth': 200,           # 最小深度(mm)
    'max_depth': 600,          # 最大深度(mm)
    'disparity_tolerance': 5,   # 视差容差(像素)
    'width_diff_threshold': 0.3, # 宽度差异阈值
    'min_matched_lines': 3,     # 最小匹配线段数
}

# ====================== 极线矫正相关函数 ======================

def stereo_rectify(left_img, right_img):
    """
    双目极线矫正（改进版本）
    Args:
        left_img: 左图
        right_img: 右图
    Returns:
        rectified_left: 矫正后的左图
        rectified_right: 矫正后的右图
        Q: 重投影矩阵
        R1: 左图旋转矩阵（用于坐标转换）
        R2: 右图旋转矩阵
    """
    # 获取图像尺寸
    h, w = left_img.shape[:2]
    
    # 计算立体矫正映射
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        CAMERA_MATRIX_LEFT, DIST_COEFF_LEFT,
        CAMERA_MATRIX_RIGHT, DIST_COEFF_RIGHT,
        (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9
    )
    
    # 计算矫正映射
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        CAMERA_MATRIX_LEFT, DIST_COEFF_LEFT, R1, P1, (w, h), cv2.CV_32FC1
    )
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        CAMERA_MATRIX_RIGHT, DIST_COEFF_RIGHT, R2, P2, (w, h), cv2.CV_32FC1
    )
    
    # 应用矫正
    left_rectified = cv2.remap(left_img, left_map1, left_map2, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_img, right_map1, right_map2, cv2.INTER_LINEAR)
    
    return left_rectified, right_rectified, Q, R1, R2

def rectify_lines(lines, R, img_size):
    """
    矫正线段坐标（改进版本）
    Args:
        lines: 原始线段列表
        R: 旋转矩阵
        img_size: 图像尺寸 (w, h)
    Returns:
        矫正后的线段列表
    """
    rectified_lines = []
    for line in lines:
        # 将点坐标转换为齐次坐标
        points = line['points'].astype(np.float32)
        homogeneous = np.column_stack(points, np.ones(len(points)))
        
        # 应用旋转矩阵变换
        homogeneous = np.dot(R, homogeneous.T).T
        rect_points = homogeneous[:, :2] / homogeneous[:, 2][:, np.newaxis]
        
        # 确保坐标在图像范围内
        rect_points[:, 0] = np.clip(rect_points[:, 0], 0, img_size[0]-1)
        rect_points[:, 1] = np.clip(rect_points[:, 1], 0, img_size[1]-1)
        
        # 创建新的线段对象
        rect_line = line.copy()
        rect_line['points'] = rect_points
        rectified_lines.append(rect_line)
    
    return rectified_lines

# ====================== 图像预处理函数 ======================

def local_contrast_enhancement(gray, config):
    """局部对比度增强"""
    h, w = gray.shape
    x_start = int(w * config['local_enhance_region'][0])
    x_end = int(w * config['local_enhance_region'][1])

    region = gray[:, x_start:x_end]
    clahe = cv2.createCLAHE(clipLimit=config['clahe_clip_local'], tileGridSize=(8,8))
    enhanced = clahe.apply(region)

    alpha, beta = config['blend_weights']
    blended = cv2.addWeighted(region, alpha, enhanced, beta, 0)
    result = gray.copy()
    result[:, x_start:x_end] = blended
    return result

def enhance_laser_channel(img, config):
    """
    激光通道增强核心算法
    Args:
        img: 输入图像（单通道或三通道）
        config: 配置参数字典
    Returns:
        增强后的三通道图像（BGR格式）
    """
    if config['laser_color'] == 'gray':
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    b, g, r = cv2.split(img)
    if config['laser_color'] == 'red':
        enhanced = cv2.addWeighted(r, 2.2, cv2.add(b, g), -1.0, 0)
    elif config['laser_color'] == 'green':
        enhanced = cv2.addWeighted(g, 2.2, cv2.add(r, b), -1.0, 0)
    else:
        enhanced = cv2.addWeighted(b, 2.2, cv2.add(r, g), -1.0, 0)
    
    return cv2.merge([enhanced, enhanced, enhanced])

def adaptive_gamma_correction(img, config):
    """
    自适应伽马校正极（局部抑制高光）
    Args:
        img: 输入BGR图像
        config: 配置参数
    Returns:
        校正后的图像（BGR格式）
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, config['specular_thresh'], 255, cv2.THRESH_BINARY)
    inv_gamma = 1.0 / config['gamma_correct']
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    corrected = cv2.LUT(img, table)
    return cv2.bitwise_and(corrected, corrected, mask=mask) + cv2.bitwise_and(img, img, mask=~mask)

def multi_scale_preprocess(img, config):
    """
    多尺度预处理流水线（核心预处理流程）
    Args:
        img: 原始输入图像
        config: 配置参数
    Returns:
        预处理后的单通道灰度图
    """
    # 1：伽马校正抑制高光
    corrected = adaptive_gamma_correction(img, config)

    # 2：转换到LAB颜色空间
    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 3：自适应直方图均衡化（CLAHE）
    clahe = cv2.createCLAHE(clipLimit=config['clahe_clip'], tileGridSize=(8,8))
    l = clahe.apply(l)

    # 4：混合模糊
    blur1 = cv2.GaussianBlur(l, config['blur_kernel'], 0)
    blur2 = cv2.medianBlur(l, 5)
    merged = cv2.addWeighted(blur1, 0.6, blur2, 0.4, 0)

    # 5：激光通道增强
    enhanced = enhance_laser_channel(merged, config)

    # 转换为灰度图后进行局部增强
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    enhanced_gray = local_contrast_enhancement(gray, config)
    
    return enhanced_gray

# ====================== 激光线检测函数 ======================

def dynamic_centroid_detection(row, config):
    """
    动态阈值质心检测算法（逐行处理）
    Args:
        row: 单行像素值数组
        config: 配置参数
    Returns:
        该行的质心坐标列表
    """
    max_val = np.max(row)
    if max_val < config['min_laser_intensity']:
        return []

    thresh = max_val * config['dynamic_thresh_ratio']
    binary = np.where(row > thresh, 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config['morph_kernel'])  
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    segments, start = [], -1
    for i, val in enumerate(closed):
        if val == 255 and start == -1:
            start = i
        elif val == 0 and start != -1:
            if i - start >= config['min_line_width']:
                segments.append((start, i-1))
            start = -1
    if start != -1 and len(closed)-start >= config['min_line_width']:
        segments.append((start, len(closed)-1))

    centers = []
    for s, e in segments:
        x = np.arange(s, e+1)
        weights = row[s:e+1]
        if np.sum(weights) == 0:
            continue
        centroid = np.sum(x * weights) / np.sum(weights)
        centers.append(int(round(centroid)))
    
    return centers

def filter_endpoints_curvature(line, config):
    """
    端点曲率过滤（消除毛刺）
    Args:
        line: 输入线段点集
        config: 配置参数
    Returns:
        过滤后的线段点集
    """
    if len(line) < 10:
        return line

    epsilon = 1e-6
    head, tail = line[:10], line[-10:]
    
    def calculate_curvature(segment):
        dx = np.gradient(segment[:,0])
        dy = np.gradient(segment[:,1])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        return np.abs(d2x * dy - dx * d2y) / ((dx**2 + dy**2)**1.5 + epsilon)

    head_curvature = calculate_curvature(head) if len(head) > 2 else 0
    tail_curvature = calculate_curvature(tail) if len(tail) > 2 else 0

    if np.mean(head_curvature) > config['max_end_curvature']:
        line = line[5:]
    if np.mean(tail_curvature) > config['max_end_curvature']:
        line = line[:-5]

    return line

def extract_line_features(line, img, label=None):
    """
    提取线段特征向量
    Args:
        line: 线段点集 (N,2)
        img: 原始图像（用于提取强度特征）
        label: 线段标签（可选）
    Returns:
        特征向量字典
    """
    if len(line) < 2:
        return None
    
    # 计算线段的起点、终点和左右端点
    x_coords = line[:, 0]
    min_idx = np.argmin(x_coords)
    max_idx = np.argmax(x_coords)
    
    left_point = line[min_idx] if min_idx < max_idx else line[max_idx]
    right_point = line[max_idx] if max_idx > min_idx else line[min_idx]
    
    # 计算线段的物理长度
    length = np.linalg.norm(left_point - right_point)
    
    # 计算方向向量
    if length > 0:
        direction = (right_point - left_point) / length
    else:
        direction = np.array([1.0, 0.0])
    
    # 计算所有点的平均位置
    avg_position = np.mean(line, axis=0)
    
    # 强度特征（从原始图像获取）
    intensities = []
    for i in range(0, len(line), 5):  # 每隔5个点采样
        pt = line[i]
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            if len(img.shape) == 3:
                intensities.append(np.mean(img[y, x]))
            else:
                intensities.append(img[y, x])
    
    if not intensities:
        return None
    
    # 曲率特征
    curvature = 0
    if len(line) >= 5:
        dx = np.gradient(line[:,0])
        dy = np.gradient(line[:,1])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        curvature = np.mean(np.abs(d2x * dy - dx * d2y) / (np.power(dx**2 + dy**2, 1.5) + 1e-6))
    
    return {
        'left_point': left_point,    # 最左点
        'right_point': right_point,  # 最右点
        'avg_position': avg_position, # 平均位置
        'direction': direction,
        'length': length,
        'mean_intensity': np.mean(intensities),
        'intensity_std': np.std(intensities),
        'curvature': curvature,
        'points': line,
        'label': label  # 存储标签
    }

def merge_broken_lines(line_groups, img):
    """
    合并被判定为同一条线的多个线段
    Args:
        line_groups: 分组后的线段索引列表
        img: 原始图像
    Returns:
        合并后的线段列表
    """
    merged_lines = []
    for indices in line_groups:
        # 收集所有点
        all_points = []
        for idx in indices:
            all_points.extend(line_groups[idx][0])  # 存储点集
        
        all_points = np.array(all_points)
        
        # 按x坐标排序
        all_points = all_points[all_points[:,0].argsort()]
        
        # 创建新线条
        merged_line = {
            'points': all_points,
            'label': line_groups[indices[0]][1]  # 继承第一个线段的标签
        }
        
        merged_line['features'] = extract_line_features(all_points, img, label=merged_line['label'])
        merged_lines.append(merged_line)
    
    return merged_lines

def enhanced_match_broken_lines(lines, img, config):
    """
    基于端点x坐标差的断线匹配（核心改进）
    Args:
        lines: 检测到的线段列表
        img: 原始图像
        config: 配置参数
    Returns:
        带标签的线段列表
    """
    if len(lines) < 2:
        return lines
    
    # 步骤1：提取特征并计算每个线段的左右端点x坐标
    features = []
    for i, line in enumerate(lines):
        feat = extract_line_features(line, img)
        if feat:
            # 为线段分配临时标签
            feat['label'] = i + 1
            features.append(feat)
        else:
            features.append(None)
    
    # 仅保留有效特征
    valid_features = [f for f in features if f is not None]
    if not valid_features:
        return lines
    
    # 步骤2：按左端点x坐标排序所有线段
    sorted_indices = sorted(range(len(valid_features)), key=lambda i: valid_features[i]['left_point'][0])
    
    # 步骤3：计算当前线段右端点与下一线段左端点的距离
    distances = []
    for i in range(1, len(sorted_indices)):
        idx1 = sorted_indices[i-1]
        idx2 = sorted_indices[i]
        dist = valid_features[idx2]['left_point'][0] - valid_features[idx1]['right_point'][0]
        distances.append(dist)
    
    # 步骤4：计算全局平均距离
    if distances:
        avg_distance = np.mean(distances)
        max_distance = avg_distance * config['breakpoint_threshold']
    else:
        max_distance = float('inf')
    
    # 步骤5：合并左端点和右端点距离小于阈值的线段
    groups = []
    current_group = [sorted_indices[0]]
    
    for i in range(1, len(sorted_indices)):
        idx_prev = sorted_indices[i-1]
        idx_current = sorted_indices[i]
        
        dist = valid_features[idx_current]['left_point'][0] - valid_features[idx_prev]['right_point'][0]
        
        if dist < max_distance and dist > 0:
            current_group.append(idx_current)
        else:
            groups.append(current_group)
            current_group = [idx_current]
    
    # 添加最后一个分组
    if current_group:
        groups.append(current_group)
    
    # 将分组转换为{索引组: 标签}的字典
    group_dict = {}
    for group in groups:
        label = valid_features[group[0]]['label']
        for idx in group:
            group_dict[valid_features[idx]['label']] = (group, label)
    
    # 步骤6：合并线并分配相同标签
    merged_lines = []
    merged_labels = set()
    
    # 处理所有组
    for group, label in group_dict.values():
        if label in merged_labels:
            continue
            
        merged_labels.add(label)
        
        # 收集组内所有点
        all_points = []
        for idx in group:
            all_points.extend(valid_features[idx]['points'])
        
        # 创建新线段
        merged_line = {
            'points': np.array(all_points),
            'label': label
        }
        
        merged_line['features'] = extract_line_features(merged_line['points'], img, label=label)
        merged_lines.append(merged_line)
    
    # 添加未合并的线段
    for i, feat in enumerate(valid_features):
        if feat['label'] not in merged_labels:
            lines = [line for line in lines if 
                     np.array_equal(line['points'], feat['points'])]
            if lines:
                merged_lines.append(lines[0])
    
    print(f"断线匹配: {len(valid_features)} => 合并后 {len(merged_lines)} 条")
    return merged_lines

def geometry_based_clustering(points, img_size, config, original_img):
    """
    基于几何约束的聚类优化（核心逻辑 + 断线标记）
    Args:
        points: 原始点集
        img_size: 图像尺寸
        config: 配置参数
        original_img: 原始图像（用于特征提取）
    Returns:
        带标签的线段列表
    """
    h, w = img_size
    mask = (points[:,0] > config['roi_padding']) & (points[:,0] < w - config['roi_padding'])
    points = points[mask]

    if len(points) < config['min_samples']:
        print(f"警告: 点太少({len(points)})，无法聚类")
        return []

    db = DBSCAN(eps=config['cluster_eps'], min_samples=config['min_samples']).fit(points)

    valid_lines = []
    for label in set(db.labels_):
        if label == -1:
            continue

        cluster = points[db.labels_ == label]
        if len(cluster) < config['min_line_length'] / 2:
            continue

        # 沿x轴排序
        sorted_cluster = cluster[cluster[:,0].argsort()]
        
        # 平滑处理
        try:
            tck, u = splprep(sorted_cluster.T, s=config['smooth_degree'])
            new_u = np.linspace(u.min(), u.max(), max(50, len(cluster)*2))
            new_points = np.column_stack(splev(new_u, tck))
        except:
            new_points = sorted_cluster

        new_points[:,0] = gaussian_filter1d(new_points[:,0], config['smooth_sigma'])
        new_points[:,1] = gaussian_filter1d(new_points[:,1], config['smooth_sigma'])

        # 端点过滤
        filtered_line = filter_endpoints_curvature(new_points, config)
        valid_lines.append(filtered_line)

    # 断线标记
    if valid_lines:
        labeled_lines = enhanced_match_broken_lines(
            [{'points': line} for line in valid_lines], 
            original_img, config
        )
        return labeled_lines
    
    return []

def detect_laser_lines(img, config):
    """激光线检测主流程"""
    preprocessed = multi_scale_preprocess(img, config)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config['morph_kernel'])
    closed = cv2.morphologyEx(preprocessed, cv2.MORPH_CLOSE, kernel, iterations=config['morph_iterations'])

    # 应用局部增强
    enhanced = local_contrast_enhancement(closed, {
        'local_enhance_region': config['local_enhance_region'],
        'clahe_clip_local': config['clahe_clip_local'],
        'blend_weights': config['blend_weights']
    })

    points = []
    for y in range(enhanced.shape[0]):
        centers = dynamic_centroid_detection(enhanced[y, :], config)
        points.extend([[x, y] for x in centers])

    if not points:
        print("未检测到激光点")
        return []

    lines = geometry_based_clustering(np.array(points), enhanced.shape, config, img)
    return lines

# ====================== 可视化与保存函数 ======================

def visualize_results(img, lines, title):
    """增强可视化（显示标签）"""
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title(f'{title} Original')

    preprocessed = multi_scale_preprocess(img, LEFT_CONFIG if 'L' in title else RIGHT_CONFIG)
    ax[1].imshow(preprocessed, cmap='gray')
    ax[1].set_title('Preprocessed')

    vis = img.copy()
    try:
        # 使用现代API
        cmap = plt.colormaps['tab20']
    except AttributeError:
        # 旧版本兼容
        cmap = plt.cm.get_cmap('tab20')
    
    # 显示线段标签
    label_counts = {}
    for line in lines:
        if 'label' not in line:
            continue
            
        color = cmap(line['label'] % 20)
        color_rgb = (np.array(color[:3]) * 255).astype(int).tolist()
        pts = line['points'].astype(int)
        cv2.polylines(vis, [pts], False, color_rgb, 2)
        
        # 在左端点显示标签
        if len(pts) > 0:
            x_min_idx = np.argmin(pts[:,0])
            start_pt = tuple(pts[x_min_idx].astype(int))
            if line['label'] in label_counts:
                label_counts[line['label']] += 1
            else:
                label_counts[line['label']] = 1
            cv2.putText(vis, f"L{line['label']}", start_pt, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    ax[2].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax[2].set_title(f'检测到 {len(label_counts)} 条线 ({len(lines)} 段)')
    plt.tight_layout()
    plt.show()

def save_labeled_lines(lines, filename):
    """保存带标签的线段数据"""
    with open(filename, 'w') as f:
        for line in lines:
            if 'label' in line:
                f.write(f"# Label: {line['label']}\n")
            else:
                f.write("# Label: unknown\n")
            np.savetxt(f, line['points'], fmt='%.2f', delimiter=',')
            f.write("\n")

# ====================== 立体匹配相关函数 ======================

def calculate_depth_range(config, baseline, focal_length):
    """
    计算深度范围
    Args:
        config: 配置参数
        baseline: 基线长度(mm)
        focal_length: 焦距(像素)
    Returns:
        (min_depth, max_depth)
    """
    return config['min_depth'], config['max_depth']

def find_candidate_matches_for_right_lines(left_lines, right_line, config, baseline, focal_length):
    """
    为右图中的单条线段寻找左图中的候选匹配（以右图为基准）
    Args:
        left_lines: 左图线段列表
        right_line: 右图单个线段
        config: 配置参数
        baseline: 基线长度
        focal_length: 焦距
    Returns:
        候选匹配线段的索引列表
    """
    # 获取右图线段中点和端点
    right_points = right_line['points']
    right_mid = np.mean(right_points, axis=0)
    right_min_x = np.min(right_points[:,0])
    right_max_x = np.max(right_points[:,0])
    
    # 计算可能的视差范围
    min_depth, max_depth = config['min_depth'], config['max_depth']
    min_disp = baseline * focal_length / max_depth
    max_disp = baseline * focal_length / min_depth
    
    candidates = []
    for left_idx, left_line in enumerate(left_lines):
        left_points = left_line['points']
        left_mid = np.mean(left_points, axis=0)
        left_min_x = np.min(left_points[:,0])
        left_max_x = np.max(left_points[:,0])
        
        # 检查y坐标对齐（极线矫正后应该在相同行）
        if abs(left_mid[1] - right_mid[1]) > config['position_tolerance']:
            continue
            
        # 检查端点x坐标差
        start_diff = abs(left_min_x - right_min_x)
        end_diff = abs(left_max_x - right_max_x)
        
        if start_diff > config['max_gap_for_matching'] or end_diff > config['max_gap_for_matching']:
            continue
            
        # 检查视差范围
        disparity = left_mid[0] - right_mid[0]
        if min_disp <= disparity <= max_disp:
            candidates.append(left_idx)
    
    return candidates

def evaluate_match_quality(left_line, right_line, left_img, right_img, config):
    """
    评估匹配质量
    Args:
        left_line: 左图线段
        right_line: 右图线段
        left_img: 左图
        right_img: 右图
        config: 配置参数
    Returns:
        匹配质量分数
    """
    # 计算端点x坐标差相似度
    left_min_x, left_max_x = np.min(left_line['points'][:,0]), np.max(left_line['points'][:,0])
    right_min_x, right_max_x = np.min(right_line['points'][:,0]), np.max(right_line['points'][:,0])
    
    start_diff = abs(left_min_x - right_min_x) 
    end_diff = abs(left_max_x - right_max_x)
    
    position_sim = 1.0 - max(0, min(1.0, (start_diff + end_diff) / (config['max_gap_for_matching'] * 2)))
    
    # 计算长度相似度
    length_left = np.sqrt(np.sum(np.diff(left_line['points'], axis=0)**2, axis=1)).sum()
    length_right = np.sqrt(np.sum(np.diff(right_line['points'], axis=0)**2, axis=1)).sum()
    
    length_ratio = min(length_left, length_right) / max(length_left, length_right, 1e-6)
    
    # 计算标签相似度
    left_label = left_line.get('label', 0)
    right_label = right_line.get('label', 0)
    label_diff = abs(left_label - right_label)
    label_sim = max(0, 1 - label_diff/10.0)  # 标签差越大，相似度越低
    
    # 计算综合评分
    score = (
        position_sim * 0.5 +
        length_ratio * 0.3 +
        label_sim * 0.2
    )
    
    return score

def calculate_line_width(line_points, img):
    """
    计算线段的平均宽度
    Args:
        line_points: 线段点集
        img: 原始图像
    Returns:
        平均宽度(像素)
    """
    if len(line_points) < 2:
        return 0
    
    widths = []
    for i in range(0, len(line_points), 5):  # 每隔5个点采样
        pt = line_points[i]
        x, y = int(pt[0]), int(pt[1])
        
        if 0 <= y < img.shape[0]:
            # 提取该行的强度
            row = img[y, :] if len(img.shape) == 2 else img[y, :, 0]
            
            # 找到强度超过20%最大值的区域
            mask = row > (np.max(row) * 0.2)
            if np.any(mask):
                # 找到连续区域
                edges = np.where(np.diff(np.concatenate([[0], mask.astype(int), [0]])))[0]
                edges = edges.reshape(-1, 2) if len(edges) % 2 == 0 else edges[:-1].reshape(-1, 2)
                
                # 计算宽度
                for start, end in edges:
                    widths.append(end - start)
    
    return np.mean(widths) if widths else 1.0  # 默认1像素

def visualize_candidate_distribution(candidate_counts):
    """
    可视化每条右图线段对应的候选匹配数量
    Args:
        candidate_counts: 每条线段对应的候选匹配数量列表
    """
    if not candidate_counts:
        return
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(candidate_counts))
    plt.bar(x, candidate_counts, color='skyblue')
    
    # 添加数量标签
    plt.title('每条右图线段的候选匹配数量')
    plt.xlabel('右图线段索引（从顶部到底部）')
    plt.ylabel('候选匹配数量')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, count in enumerate(candidate_counts):
        plt.text(i, count+0.1, str(count), ha='center')
    
    plt.tight_layout()
    plt.show()

def stereo_match(left_lines, right_lines, left_img, right_img, config, Q):
    """
    立体匹配主函数（以右图为基准）
    Args:
        left_lines: 左图检测到的线段
        right_lines: 右图检测到的线段
        left_img: 左图原始图像
        right_img: 右图原始图像
        config: 配置参数
        Q: 重投影矩阵
    Returns:
        匹配结果列表
    """
    # 计算基线长度和平均焦距
    baseline = np.linalg.norm(T)
    focal_length = (CAMERA_MATRIX_LEFT[0,0] + CAMERA_MATRIX_RIGHT[0,0]) / 2
    
    # 按y坐标对右图线段排序（从上到下）
    right_lines = sorted(right_lines, key=lambda line: np.mean(line['points'][:,1]))
    
    # 统计每条右图线段的候选匹配数量
    candidate_counts = [0] * len(right_lines)
    
    matches = []
    matched_left_labels = set()
    
    # 以右图为基准，对每条右图线段寻找左图匹配
    for right_idx, right_line in enumerate(right_lines):
        # 为当前右图线段寻找左图候选匹配
        candidates = find_candidate_matches_for_right_lines(left_lines, right_line, config, baseline, focal_length)
        candidate_counts[right_idx] = len(candidates)
        
        best_score = -1
        best_left_idx = -1
        
        # 遍历所有候选匹配，寻找最佳匹配
        for left_idx in candidates:
            # 确保左图线段尚未匹配
            left_label = left_lines[left_idx].get('label', None)
            if left_label and left_label in matched_left_labels:
                continue
                
            # 计算匹配质量
            score = evaluate_match_quality(left_lines[left_idx], right_line, left_img, right_img, config)
            
            # 应用标签连续性约束：匹配线段标签应类似
            right_label = right_line.get('label', 0)
            if abs(left_label - right_label) > 5:
                score *= 0.7  # 标签差异过大则惩罚分数
                
            if score > best_score:
                best_score = score
                best_left_idx = left_idx
        
        # 如果有有效匹配
        if best_left_idx != -1 and best_score > config['min_match_score_threshold']:
            match = {
                'right_line': right_line,
                'left_line': left_lines[best_left_idx],
                'match_score': best_score
            }
            
            # 添加匹配
            matches.append(match)
            
            # 标记左图线段已被匹配
            left_label = left_lines[best_left_idx].get('label', None)
            if left_label:
                matched_left_labels.add(left_label)
    
    # 可视化候选匹配分布
    visualize_candidate_distribution(candidate_counts)
    
    # 计算深度和视极差
    for match in matches:
        left_mid = np.mean(match['left_line']['points'], axis=0)
        right_mid = np.mean(match['right_line']['points'], axis=0)
        disparity = left_mid[0] - right_mid[0]
        
        # 使用重投影矩阵计算深度
        point_3d = np.array([[[left_mid[0], left_mid[1], disparity]]], dtype=np.float32)
        try:
            point_4d = cv2.perspectiveTransform(point_3d, Q)
            depth = point_4d[0][0][2]
        except:
            # 备选深度计算
            depth = (baseline * focal_length) / (disparity + 1e-6)
        
        # 添加计算结果
        match['disparity'] = disparity
        match['depth'] = depth
        match['left_width'] = calculate_line_width(match['left_line']['points'], left_img)
        match['right_width'] = calculate_line_width(match['right_line']['points'], right_img)
    
    return matches

def calculate_circularity(depths):
    """
    计算深度值的圆度评估指标
    Args:
        depths: 深度值列表
    Returns:
        圆度评估分数 (0-1之间，1表示完美圆形分布)
    """
    if len(depths) < 3:
        return 0
    
    # 计算深度平均值
    mean_depth = np.mean(depths)
    
    # 计算相对深度变化
    variations = [abs(d - mean_depth) for d in depths]
    max_variation = max(variations)
    
    if max_variation == 0:
        return 1.0
    
    # 圆度评估（变化越小越圆）
    circularity = 1 - min(1, np.mean(variations) / max_variation)
    
    return circularity

def visualize_matches(left_img, right_img, matches):
    """
    可视化匹配结果
    Args:
        left_img: 左图
        right_img: 右图
        matches: 匹配结果列表
    """
    if not matches:
        print("无匹配结果可显示")
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    try:
        cmap = plt.colormaps['tab20']
    except AttributeError:
        cmap = plt.cm.get_cmap('tab20')
    
    # 绘制左图
    left_vis = left_img.copy()
    for i, match in enumerate(matches):
        color_rgba = cmap(i % 20)
        color = tuple(int(c * 255) for c in color_rgba[:3])
        
        # 获取左图线段
        left_line = match['left_line']
        pts = left_line['points'].astype(int)
        cv2.polylines(left_vis, [pts], False, color, 2)
        
        # 显示标签和深度
        left_label = left_line.get('label', '?')
        depth_label = f"{match['depth']:.1f}mm"
        
        if len(pts) > 0:
            # 找到左端点
            x_min_idx = np.argmin(pts[:,0])
            start_pt = tuple(pts[x_min_idx].astype(int))
            cv2.putText(left_vis, f"L{left_label}: {depth_label}", start_pt, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    ax1.imshow(cv2.cvtColor(left_vis, cv2.COLOR_BGR2RGB))
    ax1.set_title('左图匹配结果')
    
    # 绘制右图
    right_vis = right_img.copy()
    for i, match in enumerate(matches):
        color_rgba = cmap(i % 20)
        color = tuple(int(c * 255) for c in color_rgba[:3])
        
        # 获取右图线段
        right_line = match['right_line']
        pts = right_line['points'].astype(int)
        cv2.polylines(right_vis, [pts], False, color, 2)
        
        # 显示标签
        right_label = right_line.get('label', '?')
        
        if len(pts) > 0:
            # 找到左端点
            x_min_idx = np.argmin(pts[:,0])
            start_pt = tuple(pts[x_min_idx].astype(int))
            cv2.putText(right_vis, f"R{right_label}", start_pt, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    ax2.imshow(cv2.cvtColor(right_vis, cv2.COLOR_BGR2RGB))
    ax2.set_title('右图匹配结果')
    
    plt.tight_layout()
    plt.show()

# ====================== 报告生成函数 ======================

def generate_match_report(matches):
    """生成匹配结果报告"""
    report = "立体匹配结果报告\n"
    report += "=" * 50 + "\n"
    report += f"总匹配数: {len(matches)}\n\n"
    
    if not matches:
        return report
        
    # 基本统计
    depths = [m['depth'] for m in matches]
    min_depth = np.min(depths)
    max_depth = np.max(depths)
    avg_depth = np.mean(depths)
    depth_std = np.std(depths)
    
    disparities = [m['disparity'] for m in matches]
    min_disp = np.min(disparities)
    max_disp = np.max(disparities)
    avg_disp = np.mean(disparities)
    
    match_scores = [m['match_score'] for m in matches]
    avg_score = np.mean(match_scores)
    
    circularity = calculate_circularity(depths)
    
    report += f"深度范围: {min_depth:.1f}mm - {max_depth:.1f}mm\n"
    report += f"平均深度: {avg_depth:.1f}mm ± {depth_std:.1f}mm\n"
    report += f"视差范围: {min_disp:.2f}px - {max_disp:.2f}px\n"
    report += f"平均视差: {avg_disp:.2f}px\n"
    report += f"平均匹配分数: {avg_score:.3f}\n"
    report += f"圆度评估: {circularity:.2f}/1.0\n\n"
    
    # 详细匹配信息
    report += "详细匹配信息:\n"
    report += "-" * 50 + "\n"
    report += "序号 | 左图标签 | 右图标签 | 深度(mm) | 视差(px) | 匹配分数\n"
    
    for i, m in enumerate(matches):
        l_label = m['left_line'].get('label', '?')
        r_label = m['right_line'].get('label', '?')
        
        report += f"{i+1:3} | L{l_label:6} | R{r_label:6} | {m['depth']:8.1f} | {m['disparity']:9.2f} | {m['match_score']:9.3f}\n"
    
    return report

# ====================== 主程序 ======================
if __name__ == "__main__":
    # 读取图像
    left_img = cv2.imread('31.1.bmp')
    right_img = cv2.imread('31.0.bmp')

    if left_img is None or right_img is None:
        print("错误：无法读取图像文件！请检查文件路径和名称是否正确。")
        print(f"左图路径: {'31.1.bmp'}")
        print(f"右图路径: {'31.0.bmp'}")
        exit()

    print(f"图像尺寸 - 左图: {left_img.shape}, 右图: {right_img.shape}")

    # 检测激光线
    print("\n检测左图激光线...")
    left_lines = detect_laser_lines(left_img, LEFT_CONFIG)
    print(f"左图检测到 {len(left_lines)} 条线")
    
    print("\n检测右图激光线...")
    right_lines = detect_laser_lines(right_img, RIGHT_CONFIG)
    print(f"右图检测到 {len(right_lines)} 条线")

    # 可视化检测结果
    if left_lines:
        visualize_results(left_img, left_lines, '左图检测结果')
    if right_lines:
        visualize_results(right_img, right_lines, '右图检测结果')

    # 极线矫正
    print("进行极线矫正...")
    left_rectified, right_rectified, Q, R1, R2 = stereo_rectify(left_img, right_img)
    
    # 矫正线段坐标
    print("矫正线段坐标...")
    img_size = (left_img.shape[1], left_img.shape[0])  # (宽度, 高度)
    
    if left_lines and R1 is not None:
        left_lines_rectified = rectify_lines(left_lines, R1, img_size)
    else:
        left_lines_rectified = left_lines
        print("警告: 左图线段矫正失败, 使用未矫正的线段")
    
    if right_lines and R2 is not None:
        right_lines_rectified = rectify_lines(right_lines, R2, img_size)
    else:
        right_lines_rectified = right_lines
        print("警告: 右图线段矫正失败, 使用未矫正的线段")

    # 进行立体匹配
    print("\n进行立体匹配（以右图为基准）...")
    if left_lines_rectified and right_lines_rectified:
        matches = stereo_match(left_lines_rectified, right_lines_rectified, 
                             left_rectified, right_rectified, LEFT_CONFIG, Q)
        print(f"找到 {len(matches)} 对匹配线段")
        
        # 显示匹配报告
        if matches:
            circularity = calculate_circularity([m['depth'] for m in matches])
            print(f"深度分极布圆度评估: {circularity:.2f}/1.0")
            
            # 可视化匹配结果
            visualize_matches(left_rectified, right_rectified, matches)
            
            # 生成匹配报告
            report = generate_match_report(matches)
            print("\n匹配报告:")
            print(report)
            
            # 保存报告
            with open("match_report.txt", "w") as f:
                f.write(report)
            print("报告已保存为 match_report.txt")
    else:
        print("警告：没有足够的线段进行匹配")
        matches = []
    
    # 打印简要结果
    print("\n匹配结果总结:")
    for i, match in enumerate(matches, 1):
        l_label = match['left_line'].get('label', '?')
        r_label = match['right_line'].get('label', '?')
        print(f"匹配{i:2d}: L{l_label:3} ↔ R{r_label:3} | "
              f"深度: {match['depth']:6.1f}mm | 视差: {match['disparity']:6.2f}px | "
              f"分数: {match['match_score']:.3f}")
    
    # 保存带标签的线段数据
    if left_lines:
        save_labeled_lines(left_lines, 'left_labeled_lines.csv')
    if right_lines:
        save_labeled_lines(right_lines, 'right_labeled_lines.csv')
    print("\n线段结果已保存为 left_labeled_lines.csv 和 right_labeled_lines.csv")