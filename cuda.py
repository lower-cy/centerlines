import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d

# ====================== 左图参数配置（针对灰度图优化） ======================
LEFT_CONFIG = {
    # 基础参数
    'laser_color': 'gray',       # 激光颜色类型，gray表示灰度图特殊处理
    'min_laser_intensity': 70,   # [30-100] 最低有效激光强度，值越大过滤弱光点越多
    
    # 预处理参数
    'clahe_clip': 4.0,           # [1.0-5.0] 对比度受限直方图均衡化的对比度上限，值越大增强越强但噪点越多
    'blur_kernel': (5, 5),       # [(3,3)-(9,9)] 高斯模糊核大小，值越大模糊效果越强，抑制噪声但丢失细节
    'gamma_correct': 1.0,        # [0.3-1.0] 伽马校正系数，值越小高光抑制越强（0.3强抑制，1.0无校正）
    'specular_thresh': 200,      # [150-250] 高光检测阈值，值越小更多区域被视为高光
    
      # 新增局部增强参数
    'local_enhance_region': (3/4, 1.0),  # 右侧1/3区域增强
    'clahe_clip_local': 3.0,
    'blend_weights': (0.3, 0.7),

    # 形态学参数
    'morph_kernel': (5,5),
    'morph_iterations': 2,

    # 质心检测
    'dynamic_thresh_ratio': 0.4, # [0.2-0.6] 动态阈值比例，值越大检测点越少抗噪性越强
    'min_line_width': 6,         # [3-10] 最小有效线宽（像素），过滤细碎线段
    'max_line_gap': 90,           # [3-15] 最大允许横向间隙（像素），值越大允许断裂越长
    
    # 几何约束
    'roi_padding': 15,           # [0-100] 边缘裁剪宽度（像素），值越大保留中心区域越多
    'cluster_eps': 24,         # [10-50] 聚类半径（像素），值越大合并的点越多
    'min_samples': 12,           # [5-20] 最小聚类点数，值越大排除小簇越严格
    'min_line_length': 80,       # [30-100] 有效线段最小长度（像素），过滤短线段
    
    # 后处理
    'smooth_sigma': 2.5,         # [1.0-5.0] 高斯平滑强度，值越大曲线越平滑但可能失真
    'max_end_curvature': 0.1,    # [0.1-0.5] 端点最大允许曲率，值越大允许端点弯曲越明显

    # 新增ROI检测参数
    'roi_smooth_window': 31,      # [15-61] 必须是奇数，值越大边界检测越稳定
    'roi_detect_thresh': 0.8,     # [0.5-1.2] 右边界检测阈值系数
    'shadow_blur_kernel': (7,7)   # 左图不需要虚影抑制，设为最小核
}

# ====================== 右图参数配置（针对彩色图优化） ====================== 
RIGHT_CONFIG = {
    # 基础参数
    'laser_color': 'red',        # 激光颜色类型
    'min_laser_intensity':15,   # [30-100] 最低有效激光强度，值越大过滤弱光点越多
    
    # 预处理参数
    'clahe_clip': 3.5,           # [1.0-5.0] 对比度受限直方图均衡化的对比度上限，值越大增强越强但噪点越多
    'blur_kernel': (7, 7),       # [(3,3)-(9,9)] 高斯模糊核大小，值越大模糊效果越强，抑制噪声但丢失细节
    'gamma_correct': 0.58,        # 更强的高光抑制
    'specular_thresh': 180,      # 稍高的高光检测阈值

    # 新增局部增强参数
    'local_enhance_region': (3/4, 1.0),  # 右侧1/4区域增强
    'clahe_clip_local': 4.0,
    'blend_weights': (0.2, 0.8),

     # 形态学参数
    'morph_kernel': (5,5),
    'morph_iterations':2,
    
    # 质心检测
    'dynamic_thresh_ratio': 0.45,# 更高抗噪阈值
    'min_line_width': 4,         # 允许更细的激光线
    'max_line_gap':300,           # 更严格的间隙控制
    
    # 几何约束
    'roi_padding': 30,           # 更小的边缘裁剪
    'cluster_eps': 35.0,         # 更小的聚类半径避免过合并
    'min_samples': 15,           # 更大的最小样本数
    'min_line_length': 100,      # 要求更长的有效线段
    
    # 后处理
    'smooth_sigma': 1.8,         # 较轻的平滑强度
    'max_end_curvature': 0.2,     # 更严格的曲率限制，消除毛刺

    'roi_detect_thresh': 5,      # [5-30] ROI检测灵敏度，值越大有效区域越小
    'shadow_blur_kernel': (7,7),  # [(5,5)-(15,15)] 虚影抑制核大小
    'roi_smooth_window': 5,  # 更大的窗口适应复杂场景
    'smooth_degree': 2.5,     # 适中的插值平滑度
    'roi_detect_thresh': 0.1  # 更高的检测灵敏度
}

def detect_valid_region(gray, config):
    """自动检测有效区域(水平方向)"""
    # 垂直投影分析
    col_sum = np.sum(gray, axis=0)
    # 平滑处理
    blur_sum = cv2.GaussianBlur(col_sum.astype(np.float32), (config['roi_smooth_window'], 1), 0)
    # 找右边界（从右向左寻找第一个小于均值的位置）
    mean_val = np.mean(blur_sum)
    x_end = gray.shape[1]
    for i in range(len(blur_sum)-1, 0, -1):
        if blur_sum[i] < mean_val * 0.8:  # 阈值可调
            x_end = i
            break
    return 0, x_end  # 返回有效区域的左右边界

def shadow_suppression(img, config):
    """虚影抑制（在预处理前调用）"""
    # 转换到HSV空间处理亮度通道
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = cv2.GaussianBlur(hsv[:,:,2], config['shadow_blur_kernel'], 0)
    
    # 背景估计与差分
    background = cv2.medianBlur(v, 21)
    diff = cv2.subtract(v, background)
    
    # 自适应阈值处理
    _, mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(img, img, mask=mask)

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

def selfup(min,max,final_data):
    cv2.imwrite(final_data,left_img)

def enhance_laser_channel(img, config):
    """
    激光通道增强核心算法
    Args:
        img: 输入图像（单通道或三通道）
        config: 配置参数字典
    Returns:
        增强后的三通道图像（BGR格式）
    """
    # 灰度图特殊处理
    if config['laser_color'] == 'gray':
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # 确保输入为三通道
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
    b, g, r = cv2.split(img)
    
    # 根据激光颜色增强对应通道（红色激光）
    if config['laser_color'] == 'red':
        # 增强红色通道
        enhanced = cv2.addWeighted(r, 2.2, cv2.add(b, g), -1.0, 0)
    elif config['laser_color'] == 'green':
        enhanced = cv2.addWeighted(g, 2.2, cv2.add(r, b), -1.0, 0)
    else: 
        enhanced = cv2.addWeighted(b, 2.2, cv2.add(r, g), -1.0, 0)
    
    # 归一化处理保证像素值在0-255范围
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
    # 二值化检测高光区域
    _, mask = cv2.threshold(gray, config['specular_thresh'], 255, cv2.THRESH_BINARY)
    
    # 伽马校正查找表（仅对高光区域应用）
    inv_gamma = 1.0 / config['gamma_correct']
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    
    # 应用校正并融合结果
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
    # 新增虚影抑制步骤
    if config['laser_color'] != 'gray':  # 右图专用
        img = shadow_suppression(img, config)
    
    # 原有处理流程保持不变
    corrected = adaptive_gamma_correction(img, config)
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
    # 排除低亮度行（根据min_laser_intensity）
    if max_val < config['min_laser_intensity']:
        return []
    
    # 动态阈值计算（dynamic_thresh_ratio控制灵敏度）
    thresh = max_val * config['dynamic_thresh_ratio']
    binary = np.where(row > thresh, 255, 0).astype(np.uint8)
    
    # 形态学闭合操作连接断裂（max_line_gap控制最大间隙）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (config['max_line_gap'], 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 寻找有效线段区间
    segments = []
    start = -1
    for i, val in enumerate(closed):
        if val == 255 and start == -1:
            start = i
        elif val == 0 and start != -1:
            if i - start >= config['min_line_width']:  # 过滤短线段
                segments.append((start, i-1))
            start = -1
    if start != -1 and len(closed)-start >= config['min_line_width']:
        segments.append((start, len(closed)-1))
    
    # 加权质心计算（亚像素精度）
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
    
    epsilon = 1e-6  # 防止除以零
    
    # 计算首尾各10%点的曲率
    head = line[:10]
    tail = line[-10:]
    
    # 头部曲率计算
    dx_head = np.gradient(head[:,0])
    dy_head = np.gradient(head[:,1])
    d2x_head = np.gradient(dx_head)
    d2y_head = np.gradient(dy_head)
    denominator = (dx_head**2 + dy_head**2)**1.5 + epsilon
    curvature_head = np.abs(d2x_head * dy_head - dx_head * d2y_head) / denominator
    
    # 尾部曲率计算 
    dx_tail = np.gradient(tail[:,0])
    dy_tail = np.gradient(tail[:,1])
    d2x_tail = np.gradient(dx_tail)
    d2y_tail = np.gradient(dy_tail)
    denominator = (dx_tail**2 + dy_tail**2)**1.5 + epsilon
    curvature_tail = np.abs(d2x_tail * dy_tail - dx_tail * d2y_tail) / denominator
    
    # 曲率阈值过滤（max_end_curvature控制灵敏度）
    if np.mean(curvature_head) > config['max_end_curvature']:
        line = line[5:]  # 去除头部异常点
    if np.mean(curvature_tail) > config['max_end_curvature']:
        line = line[:-5] # 去除尾部异常点
    
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
    # 边缘裁剪（roi_padding控制裁剪范围）
    mask = (points[:,0] > config['roi_padding']) & \
           (points[:,0] < w - config['roi_padding']) & \
           (points[:,1] > config['roi_padding']) & \
           (points[:,1] < h - config['roi_padding'])
    points = points[mask]
    
    # DBSCAN聚类（cluster_eps和min_samples控制聚类粒度）
    db = DBSCAN(eps=config['cluster_eps'], min_samples=config['min_samples']).fit(points)
    
    valid_lines = []
    for label in set(db.labels_):
        if label == -1:
            continue
            
        cluster = points[db.labels_ == label]
        # 过滤短线段（min_line_length）
        if len(cluster) < config['min_line_length']:
            continue
        
        # 按y坐标排序
        sorted_cluster = cluster[cluster[:,1].argsort()]
        
        # 样条插值平滑
        try:
            tck, u = splprep(sorted_cluster.T, s=config['smooth_degree'])
            new_u = np.linspace(u.min(), u.max(), int(len(u)*2))
            new_points = np.column_stack(splev(new_u, tck))
        except:
            new_points = sorted_cluster
        
        # 高斯滤波（smooth_sigma控制平滑度）
        new_points[:,0] = gaussian_filter1d(new_points[:,0], config['smooth_sigma'])
        new_points[:,1] = gaussian_filter1d(new_points[:,1], config['smooth_sigma'])
        
        # 端点曲率过滤
        filtered_line = filter_endpoints_curvature(new_points, config)
        
        valid_lines.append(filtered_line)
    
    return valid_lines
def detect_laser_lines(img, config):
    """激光线检测主流程"""
    # 预处理
    preprocessed = multi_scale_preprocess(img, config)
    
    # 自动检测有效区域
    x_start, x_end = detect_valid_region(preprocessed, config)
    roi_width = x_end - x_start
    
    # 在有效区域内进行形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config['morph_kernel'])
    closed_roi = cv2.morphologyEx(preprocessed[:, x_start:x_end], 
                                cv2.MORPH_CLOSE, kernel,
                                iterations=config['morph_iterations'])
    
    # 局部增强（仅处理右侧30%区域）
    enhanced_roi = local_contrast_enhancement(closed_roi, {
        'local_enhance_region': (0.7, 1.0),
        'clahe_clip_local': 3.0,
        'blend_weights': (0.3, 0.7)
    })
    
    # 合并回原图
    closed = preprocessed.copy()
    closed[:, x_start:x_end] = enhanced_roi
    
    # 逐行检测时只处理有效区域
    points = []
    for y in range(closed.shape[0]):
        centers = dynamic_centroid_detection(closed[y, x_start:x_end], config)
        points.extend([[x_start + x, y] for x in centers])
    
    if not points:
        return []
    
    # 几何约束聚类
    lines = geometry_based_clustering(np.array(points), closed.shape, config)
    return lines

def visualize_results(img, lines, title):
    """增强可视化"""
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始图像
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title(f'{title} Original')
    
    # 预处理效果
    preprocessed = multi_scale_preprocess(img, LEFT_CONFIG if 'L' in title else RIGHT_CONFIG)
    ax[1].imshow(preprocessed, cmap='gray')
    ax[1].set_title('Preprocessed')
    
    # 检测结果
    vis = img.copy()
    colors = [(0,255,0), (255,0,0), (0,0,255)]
    for i, line in enumerate(lines):
        color = colors[i % len(colors)]
        pts = line.astype(int)
        cv2.polylines(vis, [pts], False, color, 2)
        # 标注端点
        if len(pts) > 0:
            cv2.circle(vis, tuple(pts[0]), 5, (0,0,255), -1)
            cv2.circle(vis, tuple(pts[-1]), 5, (255,0,0), -1)
    ax[2].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax[2].set_title(f'Detected {len(lines)} Lines')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 读取图像
    left_img = cv2.imread('L.bmp')
    right_img = cv2.imread('R.jpg')
    
    # 左图处理
    print("处理左图(L.bmp)...")
    left_lines = detect_laser_lines(left_img, LEFT_CONFIG)
    print(f"左图提取到 {len(left_lines)} 条中心线")
  
    # 右图处理
    print("\n处理右图(R.jpg)...")
    right_lines = detect_laser_lines(right_img, RIGHT_CONFIG)
    print(f"右图提取到 {len(right_lines)} 条中心线")   
    visualize_results(left_img, left_lines, 'Left Image')
    visualize_results(right_img, right_lines, 'Right Image')
    
    # 保存结果
    def save_clean_lines(lines, filename):
        with open(filename, 'w') as f:
            for i, line in enumerate(lines):
                # 去除首尾可能存在的NaN值
                clean_line = line[~np.isnan(line).any(axis=1)]
                np.savetxt(f, clean_line, fmt='%.2f', 
                          header=f'Line {i+1}', comments='# ',
                          delimiter=',')
    
    save_clean_lines(left_lines, 'left_clean_lines.csv')
    save_clean_lines(right_lines, 'right_clean_lines.csv')
    print("结果已保存为 left_clean_lines.csv 和 right_clean_lines.csv")