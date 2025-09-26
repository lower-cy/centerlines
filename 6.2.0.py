# 6.1.3.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d


# ====================== 左图参数配置（针对灰度图优化） ======================
LEFT_CONFIG = {
    # 基础参数
    'laser_color': 'gray',       # 激光颜色类型
    'min_laser_intensity': 45,  # [30-100] 最低有效激光强度
    
    # 预处理参数
    'clahe_clip': 3.5,          # [1.0-5.0] 对比度增强上限
    'blur_kernel': (3, 3),      # 高斯模糊核大小
    'gamma_correct': 1.0,       # 伽马校正系数
    'specular_thresh': 200,     # 高光检测阈值
    
    # 新增局部增强参数
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
    'min_samples': 6,          # 最小样本数
    'min_line_length': 80,      # 有效线段长度

    # 后处理
    'smooth_sigma': 2.5,        # 平滑强度
    'max_end_curvature': 0.08, # 更严格的端点曲率限制
    'smooth_degree': 3.0,       # 插值平滑度
    
    # 曲线补全参数
    'max_gap_to_connect': 200,   # 最大可连接间隙（像素）
    'angle_similarity': 0.98,   # 方向相似度阈值（cosθ）
    'min_segment_length': 20,   # 可生长线段最小长度
}

# ====================== 右图参数配置（针对彩色图优化） ====================== 
RIGHT_CONFIG = {
    # 基础参数
    'laser_color': 'red',        # 激光颜色类型
    'min_laser_intensity': 45,   # 最低有效激光强度

    # 预处理参数
    'clahe_clip': 2.0,           # 对比度增强上限
    'blur_kernel': (3, 3),       # 高斯模糊核大小
    'gamma_correct': 0.75,       # 高光抑制
    'specular_thresh': 180,      # 高光检测阈值

    # 新增局部增强参数
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
    
    # 曲线补全参数
    'max_gap_to_connect': 200,    # 最大可连接间隙
    'angle_similarity': 0.95,    # 方向相似度阈值
    'min_segment_length': 15,     # 可生长线段最小长度
}


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
    自适应伽马校正（局部抑制高光）
    Args:
        img: 输入BGR图像
        config: 配置参数
    Returns:
        校正后的图像（BGR格式）
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用双边阈值抑制极端高光
    _, strong_mask = cv2.threshold(gray, config['specular_thresh']*1.2, 255, cv2.THRESH_BINARY)
    _, weak_mask = cv2.threshold(gray, config['specular_thresh'], 255, cv2.THRESH_BINARY)
    
    # 对极端高光区域应用更强的抑制
    inv_gamma = 1.0 / (config['gamma_correct'] * 1.3)  # 增强高光抑制
    strong_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    corrected = cv2.LUT(img, strong_table)
    
    # 对普通高光区域应用常规抑制
    inv_gamma = 1.0 / config['gamma_correct']
    weak_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    weak_corrected = cv2.LUT(img, weak_table)
    
    # 分层融合
    result = cv2.bitwise_and(corrected, corrected, mask=strong_mask)
    result += cv2.bitwise_and(weak_corrected, weak_corrected, mask=cv2.bitwise_not(strong_mask))
    return result


def remove_background_stripe(img, config):
    """
    方向敏感背景去除
    Args:
        img: 输入单通道图像
        config: 配置参数
    Returns:
        去噪后的图像
    """
    # 创建方向敏感结构元素（竖直方向增强）
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))  # 强化竖直结构
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))  # 水平干扰抑制
    
    # 分离背景和前景
    bg_v = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_v)
    bg_h = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_h)
    
    # 抑制水平长条纹
    bg_removed = cv2.addWeighted(bg_v, 1.0, bg_h, -0.5, 0)
    
    # 自适应融合原始图像
    alpha = 0.8 + (bg_removed.astype(float)/255.0)*0.2  # 动态权重
    result = img.astype(float) * alpha + bg_removed.astype(float)*(1-alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


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

    if np.mean(calculate_curvature(head)) > config['max_end_curvature']:
        line = line[5:]
    if np.mean(calculate_curvature(tail)) > config['max_end_curvature']:
        line = line[:-5]

    return line


def connect_broken_lines(lines, img_shape, config):
    """
    基于曲率连续性的竖向断线补全
    """
    if not lines:
        return []
    
    # 按线段中点x坐标排序
    def get_x_center(line):
        return np.mean(line[:, 0])
    lines.sort(key=lambda x: get_x_center(x))
    
    connected_lines = []
    current_line = lines[0]
    
    for i in range(1, len(lines)):
        next_line = lines[i]
        
        # 计算曲率相似度
        current_curvature = calculate_segment_curvature(current_line)
        next_curvature = calculate_segment_curvature(next_line)
        
        # 计算方向相似度
        dir_similarity = calculate_direction_similarity(current_line, next_line)
        
        # 多条件判断
        if (abs(current_curvature - next_curvature) < 0.05 and  # 曲率差异阈值
            dir_similarity > config['angle_similarity'] and 
            abs(current_line[-1][1] - next_line[0][1]) < config['max_gap_to_connect']):
            
            # 融合线段
            merged_line = merge_segments(current_line, next_line)
            current_line = merged_line
        else:
            connected_lines.append(current_line)
            current_line = next_line
            
    connected_lines.append(current_line)
    return connected_lines


def calculate_segment_curvature(segment):
    """计算线段曲率"""
    if len(segment) < 5:
        return 0
    dx = np.gradient(segment[:,0])
    dy = np.gradient(segment[:,1])
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    curvature = np.abs(d2x * dy - dx * d2y) / ((dx**2 + dy**2)**1.5 + 1e-6)
    return np.mean(curvature)


def calculate_direction_similarity(line1, line2):
    """计算两个线段方向相似度"""
    vec1 = line1[-1] - line1[0]
    vec2 = line2[-1] - line2[0]
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return np.abs(np.dot(vec1, vec2) / (norm + 1e-6))


def merge_segments(line1, line2):
    """融合两个线段"""
    return np.vstack((line1, line2))


def grow_line(line, max_gap, img_shape):
    """
    竖向线段生长算法（严格限制y轴生长）
    Args:
        line: 输入线段
        max_gap: 最大生长间隙
        img_shape: 图像尺寸
    Returns:
        竖向生长后的线段
    """
    if len(line) < 2:
        return line

    # 计算线段x中心坐标
    x_center = np.mean(line[:, 0])
    
    # 生长方向固定为y轴方向
    y_min, y_max = int(np.min(line[:,1])), int(np.max(line[:,1]))
    
    # 向两端生长
    new_points = []
    
    # 向上生长
    for i in range(max_gap):
        if y_max + i + 1 < img_shape[0]:
            new_points.append([int(x_center), y_max + i + 1])
    
    # 向下生长
    for i in range(max_gap):
        if y_min - i - 1 >= 0:
            new_points.append([int(x_center), y_min - i - 1])
    
    if new_points:
        line = np.vstack((line, np.array(new_points)))
    
    return line


# 调整后的聚类参数（增大聚类半径）
LEFT_CONFIG.update({
    'cluster_eps': 24,  # 增大聚类半径（竖向线段间有横向间隙）
    'min_samples': 10, # 提高最小样本数避免过连接
})

RIGHT_CONFIG.update({
    'cluster_eps': 12,
    'min_samples': 8,
})


def track_and_extend_lines(lines, img_shape, config):
    """
    线段跟踪与特征继承
    """
    if not lines:
        return []
    
    # 按起始位置排序
    lines.sort(key=lambda x: x[0][1])  # 按y坐标排序
    
    tracked_lines = []
    for i, line in enumerate(lines):
        extended_line = line.copy()
        
        # 获取前驱线段特征
        if i > 0:
            prev_line = lines[i-1]
            feature_vector = extract_line_features(prev_line)
            
            # 根据前驱特征延伸
            extension = extend_by_feature(prev_line, feature_vector, img_shape)
            extended_line = merge_segments(line, extension)
        
        # 应用曲率约束延伸
        extended_line = curvature_based_extension(extended_line, img_shape, config)
        tracked_lines.append(extended_line)
    
    return tracked_lines


def extract_line_features(line):
    """提取线段特征向量"""
    features = {
        'curvature': calculate_segment_curvature(line),
        'direction': line[-1] - line[0],
        'length': len(line),
        'position': np.mean(line[:,1])  # 平均y坐标
    }
    return features


def extend_by_feature(line, features, img_shape):
    """根据特征延伸线段"""
    direction = features['direction']
    length = features['length']
    
    # 计算延伸方向
    unit_dir = direction / (np.linalg.norm(direction) + 1e-6)
    
    # 计算延伸点
    new_points = []
    for i in range(int(length/2)):
        step = unit_dir * (i + 1)
        new_point = line[-1] + step.astype(int)
        if all(0 <= new_point[i] < img_shape[1-i] for i in [0,1]):
            new_points.append(new_point)
    
    return np.array(new_points) if new_points else line


def curvature_based_extension(line, shape, config):
    """基于曲率的延伸算法"""
    # 实现基于现有曲率的平滑延伸
    # 这里可以使用样条插值预测延伸路径
    try:
        tck, u = splprep(line.T, s=config['smooth_degree']*2)
        new_u = np.linspace(u.min(), u.max()*1.2, int(len(u)*1.5))
        extended = np.column_stack(splev(new_u, tck))
        return extended.astype(np.int32)
    except:
        return line


def geometry_based_clustering(points, img_size, config):
    """
    基于几何约束的聚类优化（核心逻辑 + 曲线补全）
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

    # 结构光曲线补全
    if valid_lines:
        valid_lines = connect_broken_lines(valid_lines, img_size, config)
        valid_lines = track_and_extend_lines(valid_lines, img_size, config)
    
    return valid_lines


def detect_laser_lines(img, config):
    """激光线检测主流程（增强版）"""
    preprocessed = multi_scale_preprocess(img, config)
    
    # 新增背景去除步骤
    preprocessed = remove_background_stripe(preprocessed, config)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config['morph_kernel'])
    closed = cv2.morphologyEx(preprocessed, cv2.MORPH_CLOSE, kernel, iterations=config['morph_iterations'])

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
    
    # 新增线段跟踪与延伸
    lines = track_and_extend_lines(lines, enhanced.shape, config)
    
    return lines


def visualize_results(img, lines, title):
    """增强可视化"""
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title(f'{title} Original')

    preprocessed = multi_scale_preprocess(img, LEFT_CONFIG if 'L' in title else RIGHT_CONFIG)
    ax[1].imshow(preprocessed, cmap='gray')
    ax[1].set_title('Preprocessed')

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


if __name__ == "__main__":
    left_img = cv2.imread('28.1.bmp')
    right_img = cv2.imread('28.0.bmp')

    print("处理左图(L.bmp)...")
    left_lines = detect_laser_lines(left_img, LEFT_CONFIG)
    print(f"左图提取到 {len(left_lines)} 条中心线")

    print("\n处理右图(R.jpg)...")
    right_lines = detect_laser_lines(right_img, RIGHT_CONFIG)
    print(f"右图提取到 {len(right_lines)} 条中心线")

    visualize_results(left_img, left_lines, 'Left Image')
    visualize_results(right_img, right_lines, 'Right Image')

    def save_clean_lines(lines, filename):
        """清洗并保存数据"""
        with open(filename, 'w') as f:
            for i, line in enumerate(lines):
                clean_line = line[~np.isnan(line).any(axis=1)]
                np.savetxt(f, clean_line, fmt='%.2f',
                          header=f'Line {i+1}', comments='# ', delimiter=',')

    save_clean_lines(left_lines, 'left_clean_lines.csv')
    save_clean_lines(right_lines, 'right_clean_lines.csv')
    print("结果已保存为 left_clean_lines.csv 和 right_clean_lines.csv")