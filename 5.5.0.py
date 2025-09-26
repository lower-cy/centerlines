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
    'min_laser_intensity': 140,   # [30-100] 最低有效激光强度，值越大过滤弱光点越多

    # 预处理参数
    'clahe_clip': 3.5,           # [1.0-5.0] 对比度受限直方图均衡化的对比度上限，值越大增强越强但噪点越多
    'blur_kernel': (7, 7),       # [(3,3)-(9,9)] 高斯模糊核大小，值越大模糊效果越强，抑制噪声但丢失细节
    'gamma_correct': 1.0,        # [0.3-1.0] 伽马校正系数，值越小高光抑制越强（0.3强抑制，1.0无校正）
    'specular_thresh': 200,      # [150-250] 高光检测阈值，值越小更多区域被视为高光

    # 新增局部增强参数
    'local_enhance_region': (2/3, 1),  # 右侧1/3区域增强
    'clahe_clip_local': 4.5,
    'blend_weights': (0.2, 0.8),

    # 形态学参数
    'morph_kernel': (5, 11),      # 竖向激光线闭合连接
    'morph_iterations': 4,
    # 质心检测
    'dynamic_thresh_ratio':0.6, # [0.2-0.6] 动态阈值比例，值越大检测点越少抗噪性越强
    'min_line_width': 1,         # [3-10] 最小有效线宽（像素），过滤细碎线段
    'max_line_gap': 50,           # 【减小间隙容忍度】防止误连

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
    'min_laser_intensity': 45,   # [30-100] 最低有效激光强度，值越大过滤弱光点越多

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
    'morph_kernel': (5, 11),     # 更适合检测竖向特征
    'morph_iterations': 3,

    # 质心检测
    'dynamic_thresh_ratio': 0.25,# 抗噪阈值
    'min_line_width': 4,         # 激光线宽度
    'max_line_gap': 50,          # 断裂容忍度

    # 几何约束
    'roi_padding': 15,           # 边缘裁剪
    'cluster_eps': 12,           # 聚类半径   避免过合并
    'min_samples': 8,           # 最小样本数
    'min_line_length': 100,      # 有效线段长度

    # 后处理
    'smooth_sigma': 1.5,         # 平滑强度
    'max_end_curvature': 0.2,    # 端点曲率限制，消除毛刺

    'smooth_degree': 2.5,        # 插值平滑度
    
    # 新增阴影去除参数
    'shadow_region': (2/3, 1),   # 处理图像区域
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
    
    # 转换为灰度图处理
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # 提取下侧区域
    bottom_region = gray[start_y:end_y, :]
    
    # 应用中值滤波
    median = cv2.medianBlur(bottom_region, config['median_blur_size'])
    
    # 创建阴影掩模
    _, shadow_mask = cv2.threshold(median, config['shadow_threshold'], 255, cv2.THRESH_BINARY_INV)
    
    # 如果是彩色图像，处理每个通道
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

    # 提取待增强区域
    region = gray[:, x_start:x_end]

    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=config['clahe_clip_local'], tileGridSize=(8,8))
    enhanced = clahe.apply(region)

    # 混合增强结果
    alpha, beta = config['blend_weights']
    blended = cv2.addWeighted(region, alpha, enhanced, beta, 0)

    # 合并回原图
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

    # 0：去除下侧阴影
    img = remove_shadow_region(img, config)

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