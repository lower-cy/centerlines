import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d

# ====================== 相机参数配置 ======================
CAMERA_PARAMS = {
    'focal_length': 0.9999215317777955,  # 焦距
    'baseline': 120.0,                   # 基线距离(mm)
    'pixel_size': 0.0048,                # 像素尺寸(mm)
    'min_depth': 100,                    # 最小有效深度(mm)
    'max_depth': 5000,                   # 最大有效深度(mm)
    'max_disparity': 200,                # 最大视差(像素)
    'disparity_threshold': 5.0           # 视差匹配阈值(像素)
}

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
    'min_laser_intensity': 140,   # [30-100] 最低有效激光强度，值越大过滤弱光点越多

    # 预处理参数
    'clahe_clip': 3.5,           # [1.0-5.0] 对比度受限直方图均衡化的对比度上限，值越大增强越强但噪点越多
    'blur_kernel': (7, 7),       # [(3,3)-(9,9)] 高斯模糊核大小，值越大模糊效果越强，抑制噪声但丢失细节
    'gamma_correct': 1.0,        # [0.3-1.0] 伽马校正系数，值越小高光抑制越强（0.3强抑制，1.0无校正）
    'specular_thresh':185,      # [150-250] 高光检测阈值，值越小更多区域被视为高光

    # 新增局部增强参数
    'local_enhance_region': (2/3, 1),  # 右侧1/3区域增强
    'clahe_clip_local': 4.5,
    'blend_weights': (0.2, 0.8),

    # 形态学参数
    'morph_kernel': (5, 11),      # 更适合竖向激光线闭合连接
    'morph_iterations': 4,
    # 质心检测
    'dynamic_thresh_ratio':0.6, # [0.2-0.6] 动态阈值比例，值越大检测点越少抗噪性越强
    'min_line_width': 1,         # [3-10] 最小有效线宽（像素），过滤细碎线段
    'max_line_gap': 200,           # 【减小间隙容忍度】防止误连

    # 几何约束
    'roi_padding': 10,           # [0-100] 边缘裁剪宽度（像素），值越大保留中心区域越多
    'cluster_eps': 24,           # 【更小半径】聚类半径（像素），值越大合并的点越多
    'min_samples': 8,           # [5-20] 最小聚类点数，值越大排除小簇越严格
    'min_line_length': 80,       # [30-100] 有效线段最小长度（像素），过滤短线段

    # 后处理
    'smooth_sigma': 2.5,         # 【适中平滑】高斯平滑强度，值越大曲线越平滑但可能失真
    'max_end_curvature': 0.1,    # [0.1-0.5] 端点最大允许曲率，值越大允许端点弯曲越明显
    'smooth_degree': 2.5,        # 适中的插值平滑度
    
    # 新增阴影去除参数
    'shadow_region': (2/3, 1),   # 处理图像下1/3区域
    'median_blur_size': 21,      # 中值滤波核大小
    'shadow_threshold': 30,      # 阴影阈值
}

# ====================== 右图参数配置（针对彩色图优化） ====================== 
RIGHT_CONFIG = {
    # 基础参数
    'laser_color': 'red',        # 激光颜色类型
    'min_laser_intensity': 20,   # [30-100] 最低有效激光强度，值越大过滤弱光点越多

    # 预处理参数
    'clahe_clip': 2.0,           # [1.0-5.0] 对比度受限直方图均衡化的对比度上限，值越大增强越强但噪点越多
    'blur_kernel': (5, 5),       # [(3,3)-(9,9)] 高斯模糊核大小，值越大模糊效果越强，抑制噪声但丢失细节
    'gamma_correct': 0.75,       # 更强的高光抑制
    'specular_thresh': 180,      # 稍高的高光检测阈值

    # 新增局部增强参数
    'local_enhance_region': (1/2, 3/4),  # 右侧1/4区域增强
    'clahe_clip_local': 5.0,             # 控制增强强度，数值越高增强越明显
    'blend_weights': (0.2, 0.8),         # 原图与增强图的混合比例

    # 形态学参数
    'morph_kernel': (5, 11),     # 【改为纵向】更适合检测竖向特征
    'morph_iterations': 3,

    # 质心检测
    'dynamic_thresh_ratio': 0.20,# 更高抗噪阈值
    'min_line_width': 4,         # 允许更细的激光线
    'max_line_gap': 150,          # 【缩小断裂容忍度】

    # 几何约束
    'roi_padding': 15,           # 更小的边缘裁剪
    'cluster_eps': 12,           # 更小的聚类半径避免过合并
    'min_samples': 8,           # 更大的最小样本数
    'min_line_length': 100,      # 要求更长的有效线段

    # 后处理
    'smooth_sigma': 1.5,         # 较轻的平滑强度
    'max_end_curvature': 0.2,    # 更严格的曲率限制，消除毛刺
    'smooth_degree': 2.5,        # 适中的插值平滑度
    
    # 新增阴影去除参数
    'shadow_region': (2/3, 1),   # 处理图像下1/3区域
    'median_blur_size': 21,      # 中值滤波核大小
    'shadow_threshold': 30,      # 阴影阈值
}

def remove_shadow_region(img, config):
    """
    去除图像下侧的无用阴影区域
    Args:
        img: 输入图像
        config: 配置参数
    Returns:
        处理后的图像
    """
    h, w = img.shape[:2]
    start_y = int(h * config['shadow_region'][0])
    end_y = int(h * config['shadow_region'][1])
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    bottom_region = gray[start_y:end_y, :]
    median = cv2.medianBlur(bottom_region, config['median_blur_size'])
    _, shadow_mask = cv2.threshold(median, config['shadow_threshold'], 255, cv2.THRESH_BINARY_INV)
    
    if len(img.shape) == 3:
        result = img.copy()
        for i in range(3):
            channel = result[start_y:end_y, :, i]
            channel[shadow_mask == 255] = 0
            result[start_y:end_y, :, i] = channel
    else:
        result = gray.copy()
        result[start_y:end_y, :][shadow_mask == 255] = 0
    
    return result

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

    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.merge([enhanced, enhanced, enhanced])

def adaptive_gamma_correction(img, config):
    """
    自适应伽马校正（局部抑制高光）
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
    return cv2.bitwise_and(corrected, corrected, mask=mask) + \
           cv2.bitwise_and(img, img, mask=~mask)

def multi_scale_preprocess(img, config):
    """
    多尺度预处理流水线（核心预处理流程）
    Args:
        img: 原始输入图像
        config: 配置参数
    Returns:
        预处理后的单通道灰度图
    """
    # 步骤0：去除下侧阴影
    img = remove_shadow_region(img, config)

    # 步骤1：伽马校正抑制高光
    corrected = adaptive_gamma_correction(img, config)

    # 步骤2：转换到LAB颜色空间
    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 步骤3：自适应直方图均衡化（CLAHE）
    clahe = cv2.createCLAHE(clipLimit=config['clahe_clip'], tileGridSize=(8,8))
    l = clahe.apply(l)

    # 步骤4：混合模糊
    blur1 = cv2.GaussianBlur(l, config['blur_kernel'], 0)
    blur2 = cv2.medianBlur(l, 5)
    merged = cv2.addWeighted(blur1, 0.6, blur2, 0.4, 0)

    # 步骤5：激光通道增强
    enhanced = enhance_laser_channel(merged, config)

    # 转换为灰度图后进行局部增强
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    enhanced_gray = local_contrast_enhancement(gray, config)

    return enhanced_gray

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

    head = line[:10]
    tail = line[-10:]

    dx_head = np.gradient(head[:,0])
    dy_head = np.gradient(head[:,1])
    d2x_head = np.gradient(dx_head)
    d2y_head = np.gradient(dy_head)
    curvature_head = np.abs(d2x_head * dy_head - dx_head * d2y_head) / ((dx_head**2 + dy_head**2)**1.5 + epsilon)

    dx_tail = np.gradient(tail[:,0])
    dy_tail = np.gradient(tail[:,1])
    d2x_tail = np.gradient(dx_tail)
    d2y_tail = np.gradient(dy_tail)
    curvature_tail = np.abs(d2x_tail * dy_tail - dx_tail * d2y_tail) / ((dx_tail**2 + dy_tail**2)**1.5 + epsilon)

    if np.mean(curvature_head) > config['max_end_curvature']:
        line = line[5:]
    if np.mean(curvature_tail) > config['max_end_curvature']:
        line = line[:-5]

    return line

def geometry_based_clustering(points, img_size, config):
    """
    基于几何约束的聚类优化（核心逻辑）
    Args:
        points: 原始点集
        img_size: 图像尺寸
        config: 配置参数
    Returns:
        优化后的线段列表
    """
    h, w = img_size
    mask = (points[:,0] > config['roi_padding']) & (points[:,0] < w - config['roi_padding'])
    points = points[mask]

    db = DBSCAN(eps=config['cluster_eps'], min_samples=config['min_samples']).fit(points)

    valid_lines = []
    for label in set(db.labels_):
        if label == -1:
            continue

        cluster = points[db.labels_ == label]
        if len(cluster) < config['min_line_length']:
            continue

        sorted_cluster = cluster[cluster[:,1].argsort()]

        try:
            tck, u = splprep(sorted_cluster.T, s=config['smooth_degree'])
            new_u = np.linspace(u.min(), u.max(), int(len(u)*2))
            new_points = np.column_stack(splev(new_u, tck))
        except:
            new_points = sorted_cluster

        new_points[:,0] = gaussian_filter1d(new_points[:,0], config['smooth_sigma'])
        new_points[:,1] = gaussian_filter1d(new_points[:,1], config['smooth_sigma'])

        filtered_line = filter_endpoints_curvature(new_points, config)

        valid_lines.append(filtered_line)

    return valid_lines

def detect_laser_lines(img, config):
    """激光线检测主流程"""
    preprocessed = multi_scale_preprocess(img, config)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config['morph_kernel'])
    closed = cv2.morphologyEx(preprocessed, cv2.MORPH_CLOSE, kernel, iterations=config['morph_iterations'])

    # 局部增强逻辑保留在整个图像上执行
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
        return []

    lines = geometry_based_clustering(np.array(points), enhanced.shape, config)
    return lines

def calculate_depth(disparity):
    """
    根据视差计算深度
    Args:
        disparity: 视差值(像素)
    Returns:
        深度值(mm)
    """
    if disparity == 0:
        return float('inf')
    depth = (CAMERA_PARAMS['focal_length'] * CAMERA_PARAMS['baseline']) / (disparity * CAMERA_PARAMS['pixel_size'])
    return depth

def estimate_initial_depth(left_lines, right_lines, img_shape):
    """
    使用相机参数估计初始深度范围
    Args:
        left_lines: 左图像检测到的激光线
        right_lines: 右图像检测到的激光线
        img_shape: 图像尺寸
    Returns:
        估计的深度范围 (min_depth, max_depth)
    """
    # 使用相机参数计算理论深度范围
    min_depth = CAMERA_PARAMS['min_depth']
    max_depth = CAMERA_PARAMS['max_depth']
    
    # 如果没有检测到线，返回默认范围
    if not left_lines or not right_lines:
        return min_depth, max_depth
    
    # 计算所有可能的视差
    disparities = []
    for left_line in left_lines:
        for right_line in right_lines:
            # 找到两个线段的共同y坐标范围
            left_y_min, left_y_max = np.min(left_line[:,1]), np.max(left_line[:,1])
            right_y_min, right_y_max = np.min(right_line[:,1]), np.max(right_line[:,1])
            
            # 计算重叠的y范围
            y_overlap_min = max(left_y_min, right_y_min)
            y_overlap_max = min(left_y_max, right_y_max)
            
            # 如果没有足够的重叠区域则跳过
            if y_overlap_max - y_overlap_min < 0.5 * min(len(left_line), len(right_line)):
                continue
                
            # 提取重叠区域的点
            left_mask = (left_line[:,1] >= y_overlap_min) & (left_line[:,1] <= y_overlap_max)
            right_mask = (right_line[:,1] >= y_overlap_min) & (right_line[:,1] <= y_overlap_max)
            
            left_overlap = left_line[left_mask]
            right_overlap = right_line[right_mask]
            
            # 对y坐标进行插值对齐
            try:
                # 左线插值
                left_interp = np.interp(right_overlap[:,1], left_line[:,1], left_line[:,0])
                # 计算视差
                disparities.extend(left_interp - right_overlap[:,0])
            except:
                continue
    
    # 如果有有效视差，计算深度范围
    if disparities:
        disparities = np.array(disparities)
        disparities = disparities[(disparities > 0) & (disparities < CAMERA_PARAMS['max_disparity'])]
        if len(disparities) > 0:
            depths = (CAMERA_PARAMS['focal_length'] * CAMERA_PARAMS['baseline']) / (disparities * CAMERA_PARAMS['pixel_size'])
            min_depth = max(CAMERA_PARAMS['min_depth'], np.min(depths) * 0.9)
            max_depth = min(CAMERA_PARAMS['max_depth'], np.max(depths) * 1.1)
    
    return min_depth, max_depth

def stereo_rectify(left_img, right_img):
    """
    立体校正函数
    Args:
        left_img: 左图像
        right_img: 右图像
    Returns:
        校正后的左右图像
    """
    # 计算校正变换
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        CAMERA_MATRIX_LEFT, DIST_COEFF_LEFT,
        CAMERA_MATRIX_RIGHT, DIST_COEFF_RIGHT,
        left_img.shape[:2][::-1], R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9
    )
    
    # 计算映射表
    map1x, map1y = cv2.initUndistortRectifyMap(
        CAMERA_MATRIX_LEFT, DIST_COEFF_LEFT, R1, P1,
        left_img.shape[:2][::-1], cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        CAMERA_MATRIX_RIGHT, DIST_COEFF_RIGHT, R2, P2,
        right_img.shape[:2][::-1], cv2.CV_32FC1
    )
    
    # 应用校正变换
    left_rectified = cv2.remap(left_img, map1x, map1y, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_img, map2x, map2y, cv2.INTER_LINEAR)
    
    return left_rectified, right_rectified, Q

def match_left_right_lines(left_lines, right_lines, img_shape, Q):
    """
    基于极线约束和深度限制的左右线匹配
    Args:
        left_lines: 左图像检测到的激光线
        right_lines: 右图像检测到的激光线
        img_shape: 图像尺寸
        Q: 重投影矩阵
    Returns:
        匹配成功的线对列表，每个元素为(left_line, right_line, disparity, depth)
    """
    min_depth, max_depth = estimate_initial_depth(left_lines, right_lines, img_shape)
    matched_pairs = []
    
    for left_line in left_lines:
        for right_line in right_lines:
            # 提取重叠区域
            left_y_min, left_y_max = np.min(left_line[:,1]), np.max(left_line[:,1])
            right_y_min, right_y_max = np.min(right_line[:,1]), np.max(right_line[:,1])
            y_overlap_min = max(left_y_min, right_y_min)
            y_overlap_max = min(left_y_max, right_y_max)
            
            if y_overlap_max - y_overlap_min < 0.5 * min(len(left_line), len(right_line)):
                continue

            # 获取重叠区域点集
            left_mask = (left_line[:,1] >= y_overlap_min) & (left_line[:,1] <= y_overlap_max)
            right_mask = (right_line[:,1] >= y_overlap_min) & (right_line[:,1] <= y_overlap_max)
            left_overlap, right_overlap = left_line[left_mask], right_line[right_mask]

            try:
                # 插值对齐y坐标
                left_interp_x = np.interp(right_overlap[:,1], left_line[:,1], left_line[:,0])
                left_points = np.column_stack([left_interp_x, right_overlap[:,1]])

                # 计算视差
                disparities = left_points[:,0] - right_overlap[:,0]
                disparities = disparities[(disparities > 0) & (disparities < CAMERA_PARAMS['max_disparity'])]
                
                if len(disparities) == 0:
                    continue
                    
                # 计算平均视差和深度
                disparity = np.mean(disparities)
                depth = calculate_depth(disparity)
                
                # 深度范围过滤
                if depth < min_depth or depth > max_depth:
                    continue
                    
                # 视差一致性检查
                if np.std(disparities) > CAMERA_PARAMS['disparity_threshold']:
                    continue
                    
                matched_pairs.append((left_line, right_line, disparity, depth))
                
            except Exception as e:
                continue
    
    return matched_pairs

def visualize_results(img, lines, title, depth_info=None):
    """增强可视化"""
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title(f'{title} Original')

    preprocessed = multi_scale_preprocess(img, LEFT_CONFIG if 'L' in title else RIGHT_CONFIG)
    ax[1].imshow(preprocessed, cmap='gray')
    ax[1].set_title('Preprocessed')

    vis = img.copy()
    colors = [(0,255,0), (255,0,0), (0,0,255)]#BGR
    for i, line in enumerate(lines):
        color = colors[i % len(colors)]
        pts = line.astype(int)
        cv2.polylines(vis, [pts], False, color, 2)
        
        # 显示深度信息
        if depth_info is not None and i < len(depth_info):
            mid_idx = len(pts) // 2
            pos = tuple(pts[mid_idx])
            depth_text = f"{depth_info[i][1]:.1f}mm"
            cv2.putText(vis, depth_text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            
        if len(pts) > 0:
            cv2.circle(vis, tuple(pts[0]), 5, (0,0,255), -1)
            cv2.circle(vis, tuple(pts[-1]), 5, (255,0,0), -1)
    ax[2].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax[2].set_title(f'Detected {len(lines)} Lines')

    plt.tight_layout()
    plt.show()

def save_matched_lines(matched_pairs, filename):
    """
    保存匹配的激光线及其深度信息
    Args:
        matched_pairs: match_left_right_lines()的输出
        filename: 保存文件名
    """
    with open(filename, 'w') as f:
        for i, (left_line, right_line, disparity, depth) in enumerate(matched_pairs):
            # 保存左线
            np.savetxt(f, left_line, fmt='%.2f',
                      header=f'Matched Line Pair {i+1} - Left Line | Disparity: {disparity:.2f}px | Depth: {depth:.1f}mm', 
                      comments='# ', delimiter=',')
            # 保存右线
            np.savetxt(f, right_line, fmt='%.2f',
                      header=f'Matched Line Pair {i+1} - Right Line', 
                      comments='# ', delimiter=',')
            f.write('\n')

if __name__ == "__main__":
    # 读取图像
    left_img = cv2.imread('L.bmp')
    right_img = cv2.imread('R.jpg')

    # 立体校正
    left_rect, right_rect, Q = stereo_rectify(left_img, right_img)

    print("处理左图(L.bmp)...")
    left_lines = detect_laser_lines(left_rect, LEFT_CONFIG)
    print(f"左图提取到 {len(left_lines)} 条中心线")

    print("\n处理右图(R.jpg)...")
    right_lines = detect_laser_lines(right_rect, RIGHT_CONFIG)
    print(f"右图提取到 {len(right_lines)} 条中心线")

    # 匹配左右图像中的激光线
    matched_pairs = match_left_right_lines(left_lines, right_lines, left_rect.shape, Q)
    print(f"\n成功匹配 {len(matched_pairs)} 对激光线")

    # 提取左图的深度信息用于可视化
    left_depth_info = [(pair[2], pair[3]) for pair in matched_pairs]
    
    # 可视化结果
    visualize_results(left_rect, left_lines, 'Left Image', left_depth_info)
    visualize_results(right_rect, right_lines, 'Right Image')

    # 保存匹配结果
    save_matched_lines(matched_pairs, 'matched_lines_with_depth.csv')
    print("匹配结果已保存为 matched_lines_with_depth.csv")