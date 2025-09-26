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
    'min_laser_intensity': 140,  # [30-100] 最低有效激光强度
    
    # 预处理参数
    'clahe_clip': 3.5,          # [1.0-5.0] 对比度增强上限
    'blur_kernel': (7, 7),      # 高斯模糊核大小
    'gamma_correct': 1.0,       # 伽马校正系数
    'specular_thresh': 200,     # 高光检测阈值
    
    # 新增局部增强参数
    'local_enhance_region': (0, 1),  # 右侧1/3区域增强
    'clahe_clip_local': 4.5,
    'blend_weights': (0.2, 0.8),

    # 形态学参数
    'morph_kernel': (5, 11),    # 竖向特征检测
    'morph_iterations': 4,
    
    # 质心检测
    'dynamic_thresh_ratio':0.6, # 动态阈值比例
    'min_line_width': 1,        # 最小有效线宽
    'max_line_gap': 50,         # 断裂容忍度

    # 几何约束
    'roi_padding': 10,          # 边缘裁剪
    'cluster_eps': 16,          # 更小聚类半径（适应结构光连续性）
    'min_samples': 6,          # 最小样本数
    'min_line_length': 80,      # 有效线段长度

    # 后处理
    'smooth_sigma': 2.5,        # 平滑强度
    'max_end_curvature': 0.08, # 更严格的端点曲率限制
    'smooth_degree': 3.0,       # 插值平滑度
    
    # 新增曲线补全参数
    'max_gap_to_connect': 80,   # 最大可连接间隙（像素）
    'vertical_deviation': 0.15, # 允许的最大横向偏移（相对于线段长度）
    'min_segment_length': 200,   # 可生长线段最小长度
}

# ====================== 右图参数配置（针对彩色图优化） ====================== 
RIGHT_CONFIG = {
    # 基础参数
    'laser_color': 'red',        # 激光颜色类型
    'min_laser_intensity': 45,   # 最低有效激光强度

    # 预处理参数
    'clahe_clip': 2.0,           # 对比度增强上限
    'blur_kernel': (5, 5),       # 高斯模糊核大小
    'gamma_correct': 0.75,       # 高光抑制
    'specular_thresh': 180,      # 高光检测阈值

    # 新增局部增强参数
    'local_enhance_region': (0, 1),  # 右侧1/4区域增强
    'clahe_clip_local': 5.0,
    'blend_weights': (0.2, 0.8),

    # 形态学参数
    'morph_kernel': (5, 11),     # 竖向特征检测
    'morph_iterations': 3,

    # 质心检测
    'dynamic_thresh_ratio': 0.25,# 抗噪阈值
    'min_line_width': 4,         # 激光线宽度
    'max_line_gap': 50,          # 断裂容忍度

    # 几何约束
    'roi_padding': 15,           # 边缘裁剪
    'cluster_eps': 8,            # 更小聚类半径
    'min_samples': 6,           # 更小样本数
    'min_line_length': 100,      # 有效线段长度

    # 后处理
    'smooth_sigma': 1.5,         # 平滑强度
    'max_end_curvature': 0.15,   # 端点曲率限制
    'smooth_degree': 2.5,        # 插值平滑度
    
    # 新增曲线补全参数
    'max_gap_to_connect': 60,    # 最大可连接间隙
    'vertical_deviation': 0.2,   # 允许的最大横向偏移
    'min_segment_length': 300,     # 可生长线段最小长度
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


def is_vertical_line(line, config):
    """
    判断线段是否为竖直方向
    Args:
        line: 输入线段
        config: 配置参数
    Returns:
        是否竖直方向
    """
    if len(line) < 2:
        return False
        
    # 计算线段的斜率
    x_coords = line[:, 0]
    y_coords = line[:, 1]
    
    # 计算线段总长度
    total_length = np.sqrt(np.sum((line[-1] - line[0])**2))
    if total_length < 5:
        return True  # 短线段默认为竖直方向
        
    # 计算x方向变化与总长度的比例
    x_variation = np.max(x_coords) - np.min(x_coords)
    x_ratio = x_variation / total_length
    
    return x_ratio <= config['vertical_deviation']


def grow_line(line, max_gap, img_shape):
    """
    基于竖直方向的线段生长算法
    Args:
        line: 输入线段
        max_gap: 最大生长间隙
        img_shape: 图像尺寸
    Returns:
        生长后的线段
    """
    if len(line) < 2:
        return line

    # 计算竖直方向向量（y轴方向）
    y_coords = line[:, 1]
    direction = np.array([0, 1])  # 强制竖直方向
    
    # 获取线段的平均x坐标
    x_mean = int(np.mean(line[:, 0]))
    
    # 向两端生长
    new_points = []
    
    # 向上生长
    top_y = int(np.min(y_coords))
    for i in range(1, max_gap+1):
        new_point = np.array([x_mean, top_y - i])
        if 0 <= new_point[1] < img_shape[0]:
            new_points.append(new_point)
    
    # 向下生长
    bottom_y = int(np.max(y_coords))
    for i in range(1, max_gap+1):
        new_point = np.array([x_mean, bottom_y + i])
        if 0 <= new_point[1] < img_shape[0]:
            new_points.append(new_point)
    
    if new_points:
        line = np.vstack((line, np.array(new_points)))
    
    return line


def connect_broken_lines(lines, img_shape, config):
    """
    基于竖直方向的断线补全
    Args:
        lines: 分段线段列表
        img_shape: 图像尺寸
        config: 配置参数
    Returns:
        补全后的连续线段列表
    """
    if not lines:
        return []
    
    # 仅处理竖直方向线段
    vertical_lines = [line for line in lines if is_vertical_line(line, config)]
    
    # 按线段中点x坐标排序
    vertical_lines.sort(key=lambda x: np.mean(x[:,0]) if len(x) > 0 else float('inf'))
    
    connected_lines = []
    current_line = vertical_lines[0]
    
    for i in range(1, len(vertical_lines)):
        next_line = vertical_lines[i]
        
        # 获取当前线段末端y坐标和下一线段起点y坐标
        end_y = current_line[-1][1]
        start_y = next_line[0][1]
        
        # 计算两点间y轴距离
        y_distance = abs(end_y - start_y)
        
        # 如果y轴距离在允许范围内
        if y_distance <= config['max_gap_to_connect']:
            # 计算x坐标差异
            x_diff = abs(np.mean(current_line[:,0]) - np.mean(next_line[:,0]))
            
            # 如果x坐标差异在允许范围内
            if x_diff <= config['vertical_deviation'] * max(len(current_line), len(next_line)):
                # 合并线段
                combined = np.vstack((current_line, next_line))
                
                try:
                    # 三次样条插值
                    tck, u = splprep(combined.T, s=config['smooth_degree']*3)
                    new_u = np.linspace(u.min(), u.max(), int(len(u)*2))
                    interpolated = np.column_stack(splev(new_u, tck))
                    
                    # 高斯平滑
                    interpolated[:,0] = gaussian_filter1d(interpolated[:,0], config['smooth_sigma']*1.5)
                    interpolated[:,1] = gaussian_filter1d(interpolated[:,1], config['smooth_sigma']*1.5)
                    
                    current_line = interpolated
                except:
                    current_line = combined
        else:
            # 线段断裂较大，尝试生长连接
            grown_line = grow_line(current_line, config['max_gap_to_connect'], img_shape)
            combined = np.vstack((grown_line, next_line))
            
            try:
                tck, u = splprep(combined.T, s=config['smooth_degree']*2)
                new_u = np.linspace(u.min(), u.max(), int(len(u)*1.5))
                current_line = np.column_stack(splev(new_u, tck))
            except:
                current_line = combined
        
        # 检查最小段长度
        if len(current_line) < config['min_segment_length']:
            if connected_lines:
                connected_lines[-1] = np.vstack((connected_lines[-1], current_line))
            else:
                connected_lines.append(current_line)
        else:
            connected_lines.append(current_line)
    
    # 处理未连接的非竖直方向线段
    non_vertical = [line for line in lines if not is_vertical_line(line, config)]
    connected_lines.extend(non_vertical)
    
    return connected_lines


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
    left_img = cv2.imread('25.1.bmp')
    right_img = cv2.imread('25.0.bmp')

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