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
    'min_laser_intensity': 45,   # 最低有效激光强度

    # 全局中值滤波参数
    'median_kernel_size': 5,     # 中值滤波核大小
    
    # 预处理参数
    'clahe_clip': 2.5,           # 对比度增强上限
    'blur_kernel': (5, 5),       # 高斯模糊核大小
    'gamma_correct': 1.0,        # 伽马校正系数
    'specular_thresh': 200,      # 高光检测阈值（[150-250]）

    # 形态学参数
    'morph_kernel': (5, 11),     # 竖向激光线闭合连接
    'morph_iterations': 4,
    
    # 质心检测
    'dynamic_thresh_ratio': 0.6, # 动态阈值比例
    'min_line_width': 1,         # 最小有效线宽
    'max_line_gap': 50,          # 断裂容忍度

    # 几何约束
    'roi_padding': 10,           # 边缘裁剪
    'cluster_eps': 8,            # 更小聚类半径
    'min_samples': 4,            # 更小样本数
    'min_line_length': 300,      # 有效线段长度

    # 后处理
    'smooth_sigma': 3.0,         # 增强平滑强度
    'max_end_curvature': 0.05,   # 更严格的端点曲率限制
    'smooth_degree': 4.0,        # 插值平滑度
    
    # 曲线补全参数
    'max_gap_to_connect': 100,   # 最大可连接间隙
    'angle_similarity': 0.2,    # 方向相似度阈值
}

# ====================== 右图参数配置（针对彩色图优化） ====================== 
RIGHT_CONFIG = {
    # 基础参数
    'laser_color': 'red',        # 激光颜色类型
    'min_laser_intensity': 45,   # 最低有效激光强度

    # 全局中值滤波参数
    'median_kernel_size': 5,     # 中值滤波核大小
    
    # 预处理参数
    'clahe_clip': 1.5,           # 对比度增强上限
    'blur_kernel': (5, 5),       # 高斯模糊核大小
    'gamma_correct': 0.75,       # 高光抑制
    'specular_thresh': 180,      # 高光检测阈值

    # 形态学参数
    'morph_kernel': (5, 11),     # 竖向特征检测
    'morph_iterations': 3,

    # 质心检测
    'dynamic_thresh_ratio': 0.25,# 抗噪阈值
    'min_line_width': 4,         # 激光线宽度
    'max_line_gap': 50,          # 断裂容忍度

    # 几何约束
    'roi_padding': 15,           # 边缘裁剪
    'cluster_eps': 6,            # 更小聚类半径
    'min_samples': 4,            # 更小样本数
    'min_line_length': 100,      # 有效线段长度

    # 后处理
    'smooth_sigma': 2.0,         # 平滑强度
    'max_end_curvature': 0.1,    # 端点曲率限制
    'smooth_degree': 3.0,        # 插值平滑度
    
    # 曲线补全参数
    'max_gap_to_connect': 20,    # 最大可连接间隙
    'angle_similarity': 0.2,     # 方向相似度阈值
}


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

    # 1：伽马校正抑制高光
    corrected = adaptive_gamma_correction(img, config)

    # 2：转换到LAB颜色空间
    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 3：自适应直方图均衡化（CLAHE）
    clahe = cv2.createCLAHE(clipLimit=config['clahe_clip'], tileGridSize=(8,8))
    l = clahe.apply(l)

    # 4：全局中值滤波（新增步骤）
    merged = cv2.medianBlur(l, config['median_kernel_size'])  # 使用配置参数
    
    # 5：激光通道增强
    enhanced = enhance_laser_channel(merged, config)

    # 转换为灰度图
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    
    return gray


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


def connect_broken_lines(lines, config):
    """
    结构光曲线补全算法（基于曲率连续性）
    Args:
        lines: 分段线段列表
        config: 配置参数
    Returns:
        补全后的连续线段列表
    """
    if not lines:
        return []
    
    # 按线段起始点x坐标排序
    lines.sort(key=lambda x: x[0,0] if len(x) > 0 else float('inf'))
    
    connected_lines = []
    current_line = lines[0]
    
    for i in range(1, len(lines)):
        next_line = lines[i]
        
        # 获取当前线段末端点和下一线段起点
        end_point = current_line[-1]
        start_point = next_line[0]
        
        # 计算两点间距离
        distance = np.linalg.norm(end_point - start_point)
        
        if distance <= config['max_gap_to_connect']:
            # 计算方向向量
            dir_current = current_line[-1] - current_line[-5]
            dir_next = next_line[1] - next_line[0]
            
            # 计算方向相似度（余弦相似度）
            cos_angle = np.dot(dir_current, dir_next) / (
                np.linalg.norm(dir_current) * np.linalg.norm(dir_next) + 1e-6)
            
            if abs(cos_angle) >= config['angle_similarity']:
                # 合并线段并插值补全
                combined = np.vstack((current_line, next_line))
                
                # 生成插值曲线
                try:
                    tck, u = splprep(combined.T, s=config['smooth_degree']*2)
                    new_u = np.linspace(u.min(), u.max(), int(len(u)*1.5))
                    interpolated = np.column_stack(splev(new_u, tck))
                    
                    # 高斯平滑
                    interpolated[:,0] = gaussian_filter1d(interpolated[:,0], config['smooth_sigma']*1.2)
                    interpolated[:,1] = gaussian_filter1d(interpolated[:,1], config['smooth_sigma']*1.2)
                    
                    current_line = interpolated
                except:
                    current_line = combined
        else:
            connected_lines.append(current_line)
            current_line = next_line
    
    connected_lines.append(current_line)
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
        valid_lines = connect_broken_lines(valid_lines, config)
    
    return valid_lines


def detect_laser_lines(img, config):
    """激光线检测主流程"""
    preprocessed = multi_scale_preprocess(img, config)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config['morph_kernel'])
    closed = cv2.morphologyEx(preprocessed, cv2.MORPH_CLOSE, kernel, iterations=config['morph_iterations'])

    points = []
    for y in range(preprocessed.shape[0]):
        centers = dynamic_centroid_detection(preprocessed[y, :], config)
        points.extend([[x, y] for x in centers])

    if not points:
        return []

    lines = geometry_based_clustering(np.array(points), preprocessed.shape, config)
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
        """
        将包含NaN值的二维数组清洗后保存至指定文件。
        
        Args:
            lines: 二维数组列表，每个元素为包含数值数据的numpy数组，
                    数组中可能存在沿行方向的NaN值需要过滤
            filename: 输出文件的路径名称，保存结果将覆盖现有文件
            
        Returns:
            None: 直接写入文件不返回数据，输出文件包含清洗后的数值矩阵，
                  每个矩阵块前带有#开头的行号注释
        """
        with open(filename, 'w') as f:
            for i, line in enumerate(lines):
                # 清洗当前行数据：移除包含NaN值的子数组
                # 使用布尔索引过滤掉存在缺失值的行
                clean_line = line[~np.isnan(line).any(axis=1)]
                
                # 将清洗后的数据写入文件
                # 配置参数： 
                # - 保留两位小数格式
                np.savetxt(f, clean_line, fmt='%.2f',
                          header=f'Line {i+1}', comments='# ',
                          delimiter=',')

    save_clean_lines(left_lines, 'left_clean_lines.csv')
    save_clean_lines(right_lines, 'right_clean_lines.csv')
    print("结果已保存为 left_clean_lines.csv 和 right_clean_lines.csv")