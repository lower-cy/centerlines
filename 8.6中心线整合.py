import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, SpectralClustering
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from collections import deque

# ====================== 左图参数配置 ======================
LEFT_CONFIG = {
    # 基础参数
    'laser_color': 'gray',
    'min_laser_intensity': 75,
    
    # 预处理参数
    'clahe_clip': 3.5,
    'blur_kernel': (3, 3),
    'gamma_correct': 1.0,
    'specular_thresh': 200,
    
    # 局部增强参数
    'local_enhance_region': (0, 1),
    'clahe_clip_local': 1.5,
    'blend_weights': (0.2, 0.8),

    # 形态学参数
    'morph_kernel': (5, 11),
    'morph_iterations': 4,
    
    # 质心检测
    'dynamic_thresh_ratio': 0.6,
    'min_line_width': 1,
    'max_line_gap': 200,

    # 几何约束
    'roi_padding': 10,
    'cluster_eps': 6,
    'min_samples': 6,
    'min_line_length': 80,

    # 后处理
    'smooth_sigma': 2.5,
    'max_end_curvature': 0.08,
    'smooth_degree': 3.0,
    
    # 断线匹配参数
    'max_gap_for_matching': 500,
    'direction_similarity': 0.2,
    'intensity_similarity': 0.8,
    'position_tolerance': 30,
    'min_extension_length': 50,
    'max_extension_angle': 60,

    # 亚像素拟合参数
    'gaussian_window_size': 7,
    'min_r2_for_subpixel': 0.8,
    'subpixel_refinement': True,

    # 生长算法参数
    'growing_enabled': True,
    'growing_max_iter': 100,
    'growing_step_size': 1.2,
    'growing_angle_thresh': 15,
    'growing_intensity_thresh': 0.5,
    'growing_curvature_thresh': 0.15,
    'growing_search_radius': 15,
    'growing_min_gap': 5,
    'growing_max_gap': 50,
    'growing_direction_smoothness': 0.8
}

# ====================== 右图参数配置 ====================== 
RIGHT_CONFIG = {
    # 基础参数
    'laser_color': 'red',
    'min_laser_intensity': 75,

    # 预处理参数
    'clahe_clip': 2.0,
    'blur_kernel': (3, 3),
    'gamma_correct': 0.75,
    'specular_thresh': 180,

    # 局部增强参数
    'local_enhance_region': (0, 1),
    'clahe_clip_local': 5.0,
    'blend_weights': (0.2, 0.8),

    # 形态学参数
    'morph_kernel': (5, 11),
    'morph_iterations': 4,

    # 质心检测
    'dynamic_thresh_ratio': 0.25,
    'min_line_width': 1,
    'max_line_gap': 200,

    # 几何约束
    'roi_padding': 15,
    'cluster_eps': 6,
    'min_samples': 6,
    'min_line_length': 100,

    # 后处理
    'smooth_sigma': 2.0,
    'max_end_curvature': 0.15,
    'smooth_degree': 2.5,
    
    # 断线匹配参数
    'max_gap_for_matching': 500,
    'direction_similarity': 0.2,
    'intensity_similarity': 0.75,
    'position_tolerance': 20,
    'min_extension_length': 40,
    'max_extension_angle': 60,

    # 亚像素拟合参数
    'gaussian_window_size': 7,
    'min_r2_for_subpixel': 0.8,
    'subpixel_refinement': True,

    # 生长算法参数
    'growing_enabled': True,
    'growing_max_iter': 100,
    'growing_step_size': 1.0,
    'growing_angle_thresh': 12,
    'growing_intensity_thresh': 0.6,
    'growing_curvature_thresh': 0.12,
    'growing_search_radius': 12,
    'growing_min_gap': 5,
    'growing_max_gap': 40,
    'growing_direction_smoothness': 0.85
}

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
    """激光通道增强"""
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
    """自适应伽马校正"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, config['specular_thresh'], 255, cv2.THRESH_BINARY)
    inv_gamma = 1.0 / config['gamma_correct']
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    corrected = cv2.LUT(img, table)
    return cv2.bitwise_and(corrected, corrected, mask=mask) + cv2.bitwise_and(img, img, mask=~mask)

def multi_scale_preprocess(img, config):
    """多尺度预处理流水线"""
    corrected = adaptive_gamma_correction(img, config)
    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=config['clahe_clip'], tileGridSize=(8,8))
    l = clahe.apply(l)
    blur1 = cv2.GaussianBlur(l, config['blur_kernel'], 0)
    blur2 = cv2.medianBlur(l, 5)
    merged = cv2.addWeighted(blur1, 0.6, blur2, 0.4, 0)
    enhanced = enhance_laser_channel(merged, config)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    enhanced_gray = local_contrast_enhancement(gray, config)
    return enhanced_gray

# ====================== 中心线提取核心函数 ======================
def gaussian_1d(x, amp, mu, sigma, baseline):
    """1D高斯函数定义"""
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2)) + baseline

def subpixel_gaussian_fit(x_data, y_data, init_guess, config):
    """亚像素级高斯拟合"""
    try:
        popt, pcov = curve_fit(
            f=gaussian_1d,
            xdata=x_data,
            ydata=y_data,
            p0=init_guess,
            maxfev=1000,
            bounds=([0, min(x_data), 0.5, 0], 
                   [np.inf, max(x_data), 5.0, max(y_data)])
        )  
        
        y_pred = gaussian_1d(x_data, *popt)
        
        # 改进的R²计算
        if np.var(y_data) > 1e-6:
            r2 = 1 - np.sum((y_data - y_pred)**2) / np.sum((y_data - np.mean(y_data))**2)
        else:
            r2 = 0
            
        if r2 < config['min_r2_for_subpixel']:
            return init_guess[1], r2
        return popt[1], r2
        
    except RuntimeError as e:
        print(f"拟合失败: {str(e)}")
        return init_guess[1], 0
def dynamic_centroid_detection(row, config):
    """动态阈值质心检测"""
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
    half_win = config['gaussian_window_size'] // 2
    
    for s, e in segments:
        start_win = max(s - half_win, 0)
        end_win = min(e + half_win, len(row) - 1)
        x_data = np.arange(start_win, end_win + 1)
        y_data = row[start_win:end_win + 1]
        x_segment = np.arange(s, e+1)
        weights = row[s:e+1]
        centroid = np.sum(x_segment * weights) / np.sum(weights)
        init_guess = [np.max(y_data), centroid, 1.0, np.min(y_data)]
        
        if config['subpixel_refinement']:
            subpixel_centroid, r2 = subpixel_gaussian_fit(x_data, y_data, init_guess, config)
            centers.append(float(subpixel_centroid))
        else:
            centers.append(int(round(centroid)))
    
    return centers

def filter_endpoints_curvature(line, config):
    """端点曲率过滤"""
    if len(line) < 10:
        return line

    def calculate_curvature(segment):
        dx = np.gradient(segment[:,0])
        dy = np.gradient(segment[:,1])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        return np.abs(d2x * dy - dx * d2y) / ((dx**2 + dy**2)**1.5 + 1e-6)

    head, tail = line[:10], line[-10:]
    if np.mean(calculate_curvature(head)) > config['max_end_curvature']:
        line = line[5:]
    if np.mean(calculate_curvature(tail)) > config['max_end_curvature']:
        line = line[:-5]
    return line

# ====================== 生长算法实现 ======================
def grow_center_lines(lines, img_gray, config):
    """
    中心线生长主函数
    Args:
        lines: 已有中心线列表，每个元素是Nx2的点集
        img_gray: 灰度图像
        config: 配置参数
    Returns:
        生长后的中心线列表
    """
    if not config['growing_enabled']:
        return lines
    
    # 预处理：确保所有线都是numpy数组且按y坐标排序
    processed_lines = []
    for line in lines:
        line_arr = np.array(line)
        if len(line_arr) > 1:
            line_arr = line_arr[line_arr[:,1].argsort()]
        processed_lines.append(line_arr)
    
    # 第一阶段：单条线生长
    grown_lines = []
    for line in processed_lines:
        if len(line) < 2:
            grown_lines.append(line)
            continue
            
        # 双向生长
        forward_line = grow_directionally(line, img_gray, config, direction=1)
        reverse_line = grow_directionally(line[::-1], img_gray, config, direction=-1)
        
        # 合并生长结果
        if len(reverse_line) > 0:
            grown_line = np.vstack([reverse_line[::-1], forward_line])
        else:
            grown_line = forward_line
            
        grown_lines.append(grown_line)
    
    # 第二阶段：断点连接
    connected_lines = connect_line_breaks(grown_lines, img_gray, config)
    
    return connected_lines

def grow_directionally(line, img_gray, config, direction=1):
    """
    单向生长函数
    Args:
        line: 中心线点集
        img_gray: 灰度图像
        config: 配置参数
        direction: 生长方向(1:正向, -1:反向)
    Returns:
        生长后的点集
    """
    if len(line) < 2:
        return line
    
    current_line = line.copy()
    last_intensity = img_gray[int(current_line[-1][1]), int(current_line[-1][0])]
    iterations = 0
    
    while iterations < config['growing_max_iter']:
        iterations += 1
        last_point = current_line[-1]
        
        # 计算当前生长方向
        if len(current_line) >= 2:
            dir_vec = current_line[-1] - current_line[-2]
            dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-6)
        else:
            dir_vec = np.array([1.0, 0.0])
        
        # 生成候选生长点
        best_point, best_energy = None, -np.inf
        for angle in np.linspace(-config['growing_angle_thresh'], 
                                config['growing_angle_thresh'], 5):
            rad = np.deg2rad(angle)
            rot_mat = np.array([
                [np.cos(rad), -np.sin(rad)],
                [np.sin(rad), np.cos(rad)]
            ])
            new_dir = rot_mat @ dir_vec
            new_point = last_point + new_dir * config['growing_step_size']
            
            # 计算能量值
            energy = calculate_growing_energy(new_point, new_dir, last_intensity, current_line, img_gray, config)
            if energy > best_energy:
                best_energy = energy
                best_point = new_point
        
        # 判断是否继续生长
        if best_point is None or best_energy < config['growing_intensity_thresh']:
            break
            
        # 添加新点并更新状态
        current_line = np.vstack([current_line, best_point])
        last_intensity = img_gray[int(best_point[1]), int(best_point[0])]
        
        # 检查曲率变化
        if len(current_line) >= 3:
            dx = np.gradient(current_line[-3:,0])
            dy = np.gradient(current_line[-3:,1])
            d2x = np.gradient(dx)
            d2y = np.gradient(dy)
            curvature = np.abs(d2x[-1] * dy[-1] - dx[-1] * d2y[-1]) / ((dx[-1]**2 + dy[-1]**2)**1.5 + 1e-6)
            if curvature > config['growing_curvature_thresh']:
                break
    
    return current_line

def calculate_growing_energy(point, direction, last_intensity, current_line, img_gray, config):
    """
    计算生长能量函数
    Args:
        point: 候选点坐标
        direction: 生长方向
        last_intensity: 上一个点的强度
        current_line: 当前生长中的线
        img_gray: 灰度图像
        config: 配置参数
    Returns:
        能量值
    """
    x, y = int(round(point[0])), int(round(point[1]))
    if not (0 <= y < img_gray.shape[0] and 0 <= x < img_gray.shape[1]):
        return -np.inf
    
    # 强度衰减惩罚
    current_intensity = img_gray[y, x]
    intensity_ratio = current_intensity / (last_intensity + 1e-6)
    intensity_term = np.clip(intensity_ratio, 0, 1.5)
    
    # 方向一致性奖励
    if len(current_line) >= 2:
        last_dir = current_line[-1] - current_line[-2]
        dir_sim = np.dot(direction, last_dir) / (np.linalg.norm(direction) * np.linalg.norm(last_dir) + 1e-6)
        direction_term = max(0, dir_sim)
    else:
        direction_term = 1.0
        
    return intensity_term * 0.7 + direction_term * 0.3

def connect_line_breaks(lines, img_gray, config):
    """
    连接断裂的中心线
    Args:
        lines: 中心线列表
        img_gray: 灰度图像
        config: 配置参数
    Returns:
        连接后的中心线列表
    """
    if len(lines) <= 1:
        return lines
    
    # 计算所有线段的端点
    endpoints = []
    for i, line in enumerate(lines):
        if len(line) < 2:
            continue
        endpoints.append({
            'line_idx': i,
            'point': line[0],
            'type': 'start',
            'direction': line[1] - line[0]
        })
        endpoints.append({
            'line_idx': i,
            'point': line[-1],
            'type': 'end',
            'direction': line[-1] - line[-2]
        })
    
    # 寻找最佳连接对
    connections = []
    used_indices = set()
    
    for i in range(len(endpoints)):
        if i in used_indices:
            continue
            
        best_match = None
        best_score = -np.inf
        
        for j in range(i+1, len(endpoints)):
            if j in used_indices:
                continue
                
            # 计算连接得分
            score = calculate_connection_score(endpoints[i], endpoints[j], img_gray, config)
            if score > best_score:
                best_score = score
                best_match = j
        
        if best_match is not None and best_score > 0:
            connections.append((i, best_match))
            used_indices.add(i)
            used_indices.add(best_match)
    
    # 执行连接操作
    connected_lines = []
    line_merged = [False] * len(lines)
    
    for i, j in connections:
        ep1, ep2 = endpoints[i], endpoints[j]
        
        # 确保连接的是不同线段
        if ep1['line_idx'] == ep2['line_idx']:
            continue
            
        line1 = lines[ep1['line_idx']]
        line2 = lines[ep2['line_idx']]
        
        # 确定连接顺序
        if ep1['type'] == 'end' and ep2['type'] == 'start':
            connected_line = np.vstack([line1, line2])
        elif ep1['type'] == 'start' and ep2['type'] == 'end':
            connected_line = np.vstack([line2, line1])
        elif ep1['type'] == 'end' and ep2['type'] == 'end':
            connected_line = np.vstack([line1, line2[::-1]])
        else:  # start-start
            connected_line = np.vstack([line1[::-1], line2])
        
        connected_lines.append(connected_line)
        line_merged[ep1['line_idx']] = True
        line_merged[ep2['line_idx']] = True
    
    # 添加未连接的线段
    for i in range(len(lines)):
        if not line_merged[i] and len(lines[i]) >= 2:
            connected_lines.append(lines[i])
    
    return connected_lines

def calculate_connection_score(ep1, ep2, img_gray, config):
    """
    计算两个端点的连接得分
    Args:
        ep1: 端点1信息
        ep2: 端点2信息
        img_gray: 灰度图像
        config: 配置参数
    Returns:
        连接得分
    """
    # 计算距离
    dist = np.linalg.norm(ep1['point'] - ep2['point'])
    if dist < config['growing_min_gap'] or dist > config['growing_max_gap']:
        return -np.inf
    
    # 方向一致性
    dir_sim = np.dot(ep1['direction'], ep2['direction']) / \
              (np.linalg.norm(ep1['direction']) * np.linalg.norm(ep2['direction']) + 1e-6)
    
    # 强度连续性
    intensity1 = img_gray[int(ep1['point'][1]), int(ep1['point'][0])]
    intensity2 = img_gray[int(ep2['point'][1]), int(ep2['point'][0])]
    intensity_diff = abs(intensity1 - intensity2) / 255.0
    
    # 综合得分
    score = (0.5 * (1 - dist/config['growing_max_gap']) + 
            0.3 * max(0, dir_sim) + 
            0.2 * (1 - intensity_diff))
    
    return score

# ====================== 主处理流程 ======================
def geometry_based_clustering(points, img_size, config, original_img):
    """基于几何约束的聚类"""
    h, w = img_size
    mask = (points[:,0] > config['roi_padding']) & (points[:,0] < w - config['roi_padding'])
    points = points[mask]

    db = DBSCAN(eps=config['cluster_eps'], min_samples=config['min_samples']).fit(points)

    valid_lines = []
    for label in set(db.labels_):
        if label == -1:
            continue

        cluster = points[db.labels_ == label]
        if len(cluster) < config['min_line_length']/2:
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

    # 初始中心线提取
    lines = geometry_based_clustering(np.array(points), enhanced.shape, config, img)
    
    # 应用生长算法
    if config['growing_enabled']:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lines = grow_center_lines(lines, img_gray, config)
    
    return lines

# ====================== 可视化与保存 ======================
def visualize_results(img, lines, title):
    """结果可视化"""
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title(f'{title} Original')

    preprocessed = multi_scale_preprocess(img, LEFT_CONFIG if 'L' in title else RIGHT_CONFIG)
    ax[1].imshow(preprocessed, cmap='gray')
    ax[1].set_title('Preprocessed')

    vis = img.copy()
    colors = plt.cm.get_cmap('tab20', 20)
    unique_labels = set(range(len(lines))) if lines else set()
    
    for i, line in enumerate(lines):
        color = colors(i % 20)
        color_rgb = (np.array(color[:3]) * 255).astype(int).tolist()
        pts = line.astype(int)
        cv2.polylines(vis, [pts], False, color_rgb, 2)
        if len(pts) > 0:
            cv2.putText(vis, str(i+1), tuple(pts[0]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    ax[2].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax[2].set_title(f'Detected {len(unique_labels)} Lines')
    plt.tight_layout()
    plt.show()

def save_labeled_lines(lines, filename):
    """保存中心线数据"""
    with open(filename, 'w') as f:
        for i, line in enumerate(lines):
            f.write(f"# Line {i+1}\n")
            np.savetxt(f, line, fmt='%.2f', delimiter=',')
            f.write("\n")

if __name__ == "__main__":
    # 加载测试图像
    left_img = cv2.imread('30.1.bmp')
    right_img = cv2.imread('30.0.bmp')

    # 处理左图
    print("处理左图(L.bmp)...")
    left_lines = detect_laser_lines(left_img, LEFT_CONFIG)
    print(f"左图提取到 {len(left_lines)} 条中心线")

    # 处理右图
    print("\n处理右图(R.jpg)...")
    right_lines = detect_laser_lines(right_img, RIGHT_CONFIG)
    print(f"右图提取到 {len(right_lines)} 条中心线")

    # 可视化结果
    visualize_results(left_img, left_lines, 'Left Image')
    visualize_results(right_img, right_lines, 'Right Image')

    # 保存结果
    save_labeled_lines(left_lines, 'left_labeled_lines.csv')
    save_labeled_lines(right_lines, 'right_labeled_lines.csv')
    print("结果已保存为 left_labeled_lines.csv 和 right_labeled_lines.csv")