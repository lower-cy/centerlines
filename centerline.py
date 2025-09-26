import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d

# ====================== 左图参数配置（灰度图优化） ======================
LEFT_CONFIG = {
    # 基础参数
    'laser_color': 'gray',       # [gray/red/green/blue] 激光颜色类型
    'min_laser_intensity': 50,   # [30-100] 最低有效激光强度，值越大过滤弱光点越多
    
    # 预处理参数
    'clahe_clip': 2.5,           # [1.0-5.0] CLAHE对比度限制，值越大增强越强
    'blur_kernel': (5, 5),       # [(3,3)-(9,9)] 高斯模糊核尺寸，大尺寸抑制噪声更好
    'gamma_correct': 0.2,        # [0.6-1.0] 伽马校正系数，小值抑制高光更强
    'specular_thresh': 220,      # [200-250] 高光检测阈值，值小检测更多高光区域
    
    # 局部增强参数
    'local_enhance_region': (3/4, 1.0),  # 增强区域(x_start, x_end)比例
    'clahe_clip_local': 3.0,     # [2.0-4.0] 局部CLAHE对比度限制
    'blend_weights': (0.2, 0.8), # 原图与增强图的混合权重
    
    # 形态学参数
    'morph_kernel': (7,7),       # [(3,3)-(9,9)] 形态学操作核尺寸
    'morph_iterations': 1,       # [1-5] 形态学操作迭代次数
    'roi_detect_thresh': 0.8,     # [0.5-1.2] 右边界检测阈值系数
    
    # 质心检测
    'dynamic_thresh_ratio':0.3, # [0.3-0.6] 动态阈值比例，大值减少误检
    'min_line_width': 1,         # [3-10] 最小有效线宽（像素）
    'max_line_gap': 90,          # [50-150] 最大允许横向间隙（像素）
    
    # 几何约束
    'roi_padding': 15,           # [0-100] 边缘裁剪宽度（像素）
    'cluster_eps': 20,           # [15-40] DBSCAN聚类半径（像素）
    'min_samples': 10,           # [8-20] 最小聚类点数
    'min_line_length': 100,       # [50-150] 有效线段最小长度（像素）
    
    # 后处理
    'smooth_sigma': 2.5,         # [1.0-4.0] 高斯平滑强度
    'max_end_curvature': 0.1,    # [0.05-0.3] 端点最大允许曲率
    
    # ROI检测
    'roi_smooth_window': 17,     # [15-61] 边界检测窗口大小（必须奇数）
    'shadow_blur_kernel': (5,5),  # [(5,5)-(15,15)] 虚影抑制核尺寸
    'roi_detect_thresh': 0.1    # ROI灵敏度
}

# ====================== 右图参数配置（彩色图优化） ======================
RIGHT_CONFIG = {
    # 基础参数
    'laser_color': 'red',        # [gray/red/green/blue] 激光颜色类型
    'min_laser_intensity': 75,   # [50-100] 最低有效激光强度
    
    # 预处理参数
    'clahe_clip': 2.5,           # [1.0-4.0] CLAHE对比度限制
    'blur_kernel': (5, 5),       # [(3,3)-(7,7)] 高斯模糊核尺寸
    'gamma_correct': 0.4,        # [0.3-0.6] 伽马校正系数
    'specular_thresh': 150,      # [120-180] 高光检测阈值
    
    # 局部增强参数
    'local_enhance_region': (1/3, 1.0),  # 增强区域(x_start, x_end)比例
    'clahe_clip_local': 4.0,     # [3.0-5.0] 局部CLAHE对比度限制 
    'blend_weights': (0.2, 0.8), # 原图与增强图的混合权重
    
    # 形态学参数
    'morph_kernel': (3,3),       # [(3,3)-(5,5)] 形态学操作核尺寸
    'morph_iterations': 2,       # [2-4] 形态学操作迭代次数
    'roi_smooth_window': 21,      # 新增行（使用奇数，建议值15-61）
    
    # 质心检测
    'dynamic_thresh_ratio': 0.35,# [0.3-0.5] 动态阈值比例
    'min_line_width': 1,         # [3-6] 最小有效线宽
    'max_line_gap': 300,         # [200-400] 最大允许横向间隙
    
    # 几何约束
    'roi_padding': 30,           # [20-50] 边缘裁剪宽度
    'cluster_eps': 35.0,         # [25-45] DBSCAN聚类半径
    'min_samples': 15,           # [10-20] 最小聚类点数
    'min_line_length': 100,      # [80-150] 有效线段最小长度
    
    # 后处理
    'smooth_sigma': 1.8,         # [1.0-3.0] 高斯平滑强度
    'max_end_curvature': 0.2,    # [0.1-0.3] 端点最大曲率
    'smooth_degree': 2.5,        # [1.0-4.0] 样条平滑度
    
    # 虚影抑制
    'shadow_blur_kernel': (7,7), # [(5,5)-(9,9)] 虚影抑制核尺寸
    'roi_detect_thresh': 0.1     # [0.05-0.2] ROI检测灵敏度
}

def detect_valid_region(gray, config):
    """优化后的有效区域检测"""
    col_sum = np.sum(gray, axis=0)
    blur_sum = cv2.GaussianBlur(col_sum.astype(np.float32), 
                              (config['roi_smooth_window'], 1), 0)
    
    # 动态阈值检测
    mean_val = np.mean(blur_sum)
    adaptive_thresh = mean_val * config['roi_detect_thresh']
    
    x_end = gray.shape[1]
    for i in range(len(blur_sum)-1, 0, -1):
        if blur_sum[i] < adaptive_thresh:
            x_end = i
            break
    return 0, x_end

def shadow_suppression(img, config):
    """改进的双模式虚影抑制"""
    if config['laser_color'] == 'gray':
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        v_channel = cv2.GaussianBlur(yuv[:,:,0], 
                                   config['shadow_blur_kernel'], 0)
        background = cv2.medianBlur(v_channel, 15)
    else:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v_channel = cv2.GaussianBlur(hsv[:,:,2],
                                   config['shadow_blur_kernel'], 0)
        background = cv2.medianBlur(v_channel, 21)  # 优化核大小
    
    diff = cv2.subtract(v_channel, background)
    
    # 自适应阈值策略
    if config['laser_color'] == 'gray':
        _, mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)  # 提高阈值
    else:
        valid_diff = diff[diff > 0]
        if len(valid_diff) > 0:
            thresh = np.percentile(valid_diff, 65)  # 降低百分比阈值
        else:
            thresh = 15
        _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    
    return cv2.bitwise_and(img, img, mask=mask)

def local_contrast_enhancement(gray, config):
    """动态混合的局部增强"""
    h, w = gray.shape
    x_start = int(w * config['local_enhance_region'][0])
    x_end = int(w * config['local_enhance_region'][1])
    
    region = gray[:, x_start:x_end]
    clahe = cv2.createCLAHE(
        clipLimit=config['clahe_clip_local'], 
        tileGridSize=(8,8)
    )
    enhanced = clahe.apply(region)
    
    # 动态混合系数
    region_mean = np.mean(region)
    alpha = max(0.2, 1 - region_mean/180)  # 根据区域亮度自动调整
    beta = 1 - alpha
    
    blended = cv2.addWeighted(region, alpha, enhanced, beta, 0)
    result = gray.copy()
    result[:, x_start:x_end] = blended
    return result

def enhance_laser_channel(img, config):
    """改进的激光通道增强"""
    if config['laser_color'] == 'gray':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    b, g, r = cv2.split(img)
    
    # 动态增强系数
    if config['laser_color'] == 'red':
        enhanced = cv2.addWeighted(r, 2.0, cv2.add(b, g), -0.8, 10)  # 降低抑制强度
    elif config['laser_color'] == 'green':
        enhanced = cv2.addWeighted(g, 2.0, cv2.add(r, b), -0.8, 10)
    else:
        enhanced = cv2.addWeighted(b, 2.0, cv2.add(r, g), -0.8, 10)
    
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.merge([enhanced, enhanced, enhanced])

def adaptive_gamma_correction(img, config):
    """改进的自适应伽马校正"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, config['specular_thresh'], 255, cv2.THRESH_BINARY)
    
    # 创建基于灰度级的gamma表（修正维度问题）
    hist = cv2.calcHist([gray], [0], mask, [256], [0,256])
    is_highlight = (hist > 0).flatten()
    
    gamma_table = np.zeros(256, dtype=np.float32)
    for i in range(256):
        if is_highlight[i]:
            gamma_table[i] = config['gamma_correct']
        else:
            gamma_table[i] = 1.0
    
    # 创建查找表
    table = np.array([((i / 255.0) ** (1.0/gamma_table[i])) * 255
                     for i in range(256)]).astype("uint8")
    
    return cv2.LUT(img, table)

def multi_scale_preprocess(img, config):
    """增强的预处理流水线"""
    img = shadow_suppression(img, config)
    corrected = adaptive_gamma_correction(img, config)

    # 细节保留增强
    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    l_channel = lab[:,:,0]
    l_3channel = cv2.merge([l_channel, l_channel, l_channel])
    detail_enhanced = cv2.detailEnhance(l_3channel, 
                                      sigma_s=8,  # 降低空间参数
                                      sigma_r=0.1)  # 降低范围参数
    enhanced_l = cv2.cvtColor(detail_enhanced, cv2.COLOR_BGR2GRAY)
    lab[:,:,0] = cv2.addWeighted(l_channel, 0.6, 
                               enhanced_l, 0.4, 0)
    
    # CLAHE增强
    clahe = cv2.createCLAHE(clipLimit=config['clahe_clip'], 
                          tileGridSize=(8,8))
    l_enhanced = clahe.apply(lab[:,:,0])
    
    # 混合去噪
    blur1 = cv2.GaussianBlur(l_enhanced, 
                           config['blur_kernel'], 0)
    blur2 = cv2.medianBlur(l_enhanced, 3)  # 减小中值滤波核
    merged = cv2.addWeighted(blur1, 0.7, blur2, 0.3, 0)
    
    # 激光通道增强
    enhanced = enhance_laser_channel(merged, config)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    return local_contrast_enhancement(gray, config)

def dynamic_centroid_detection(row, config):
    """改进的动态质心检测"""
    smoothed = cv2.GaussianBlur(row, (5,1), 0)
    max_val = np.max(smoothed)
    if max_val < config['min_laser_intensity']:
        return []
    
    # 自适应基线计算
    baseline = np.median(smoothed) * 0.7
    dynamic_range = max(max_val - baseline, 1)
    thresh = baseline + dynamic_range * config['dynamic_thresh_ratio']
    
    binary = np.where(smoothed > thresh, 255, 0).astype(np.uint8)
    
    # 动态形态学核
    kernel_size = min(config['max_line_gap'], 50)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                     (kernel_size, 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 线段检测优化
    segments = []
    in_segment = False
    start_idx = -1
    for i, val in enumerate(closed):
        if val == 255 and not in_segment:
            start_idx = i
            in_segment = True
        elif val == 0 and in_segment:
            if i - start_idx >= config['min_line_width']:
                segments.append((start_idx, i-1))
            in_segment = False
            start_idx = -1
    if in_segment and len(closed)-start_idx >= config['min_line_width']:
        segments.append((start_idx, len(closed)-1))
    
    # 亚像素质心计算
    centers = []
    for s, e in segments:
        weights = smoothed[s:e+1]
        if np.sum(weights) == 0:
            continue
        x_coords = np.arange(s, e+1)
        centroid = np.sum(x_coords * weights) / np.sum(weights)
        centers.append(int(round(centroid)))
    
    return centers

def filter_endpoints_curvature(line, config):
    """优化的端点曲率过滤"""
    if len(line) < 10:
        return line
    
    epsilon = 1e-6
    check_points = 8  # 检查端点附近8个点
    
    # 头部曲率计算
    head = line[:check_points]
    dx = np.gradient(head[:,0])
    dy = np.gradient(head[:,1])
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    curvature = np.abs(d2x*dy - dx*d2y) / (dx**2 + dy**2 + epsilon)**1.5
    
    if np.mean(curvature) > config['max_end_curvature']:
        line = line[check_points//2:]  # 去除前50%异常点
    
    # 尾部曲率计算
    tail = line[-check_points:]
    dx = np.gradient(tail[:,0])
    dy = np.gradient(tail[:,1])
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    curvature = np.abs(d2x*dy - dx*d2y) / (dx**2 + dy**2 + epsilon)**1.5
    
    if np.mean(curvature) > config['max_end_curvature']:
        line = line[:-(check_points//2)]  # 去除后50%异常点
    
    return line

def geometry_based_clustering(points, img_size, config):
    """优化的几何聚类算法"""
    h, w = img_size
    mask = (points[:,0] > config['roi_padding']) & \
           (points[:,0] < w - config['roi_padding']) & \
           (points[:,1] > config['roi_padding'])
    points = points[mask]
    
    # 改进的DBSCAN参数
    db = DBSCAN(eps=config['cluster_eps'], 
              min_samples=config['min_samples'],
              metric='euclidean').fit(points)
    
    valid_lines = []
    for label in np.unique(db.labels_):
        if label == -1:
            continue
            
        cluster = points[db.labels_ == label]
        if len(cluster) < config['min_line_length']:
            continue
        
        # 改进的插值方法
        try:
            sorted_cluster = cluster[cluster[:,1].argsort()]
            tck, u = splprep(sorted_cluster.T, 
                           s=config['smooth_degree'], 
                           k=2)  # 使用二次样条
            new_u = np.linspace(u.min(), u.max(), len(cluster)*2)
            new_points = np.column_stack(splev(new_u, tck))
        except:
            new_points = sorted_cluster
        
        # 自适应高斯滤波
        sigma = min(config['smooth_sigma'], len(new_points)/10)
        new_points[:,0] = gaussian_filter1d(new_points[:,0], sigma)
        new_points[:,1] = gaussian_filter1d(new_points[:,1], sigma)
        
        filtered_line = filter_endpoints_curvature(new_points, config)
        valid_lines.append(filtered_line)
    
    return valid_lines

def detect_laser_lines(img, config):
    """优化的激光线检测主流程"""
    preprocessed = multi_scale_preprocess(img, config)
    x_start, x_end = detect_valid_region(preprocessed, config)
    
    # 改进的形态学处理
    roi = preprocessed[:, x_start:x_end]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                     config['morph_kernel'])
    processed_roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel,
                                   iterations=config['morph_iterations'])
    
    # 动态局部增强
    enhanced_roi = local_contrast_enhancement(processed_roi, {
        'local_enhance_region': config['local_enhance_region'],
        'clahe_clip_local': config['clahe_clip_local'],
        'blend_weights': config['blend_weights']
    })
    
    merged = preprocessed.copy()
    merged[:, x_start:x_end] = enhanced_roi
    
    # 并行化逐行检测
    points = []
    for y in range(merged.shape[0]):
        row = merged[y, x_start:x_end]
        centers = dynamic_centroid_detection(row, config)
        points.extend([[x_start + x, y] for x in centers])
    
    if not points:
        return []
    
    lines = geometry_based_clustering(np.array(points), 
                                    merged.shape, config)
    return lines

def visualize_results(img, lines, title):
    """增强的可视化函数"""
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    
    # 原始图像
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title(f'{title} Original')
    ax[0].axis('off')
    
    # 预处理效果
    preprocessed = multi_scale_preprocess(img, 
                   LEFT_CONFIG if 'L' in title else RIGHT_CONFIG)
    ax[1].imshow(preprocessed, cmap='gray')
    ax[1].set_title('Preprocessed')
    ax[1].axis('off')
    
    # 检测结果可视化
    vis = img.copy()
    colors = [(0,255,0), (0,0,255), (255,0,0)]
    for i, line in enumerate(lines):
        color = colors[i % len(colors)]
        pts = line.astype(int)
        cv2.polylines(vis, [pts], False, color, 2, lineType=cv2.LINE_AA)
        
        # 绘制方向箭头
        if len(pts) > 10:
            start = tuple(pts[0])
            end = tuple(pts[-1])
            cv2.arrowedLine(vis, start, end, (255,255,0), 2, 
                          tipLength=0.1)
    
    ax[2].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax[2].set_title(f'Detected {len(lines)} Laser Lines')
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def save_clean_lines(lines, filename):
    """改进的结果保存函数"""
    with open(filename, 'w') as f:
        for i, line in enumerate(lines):
            clean_line = line[~np.isnan(line).any(axis=1)]
            np.savetxt(f, clean_line, fmt='%.2f',
                      header=f'Line_{i+1}_Points',
                      comments='# ',
                      delimiter=',')
            f.write('\n')  # 添加空行分隔不同线段

if __name__ == "__main__":
    # 图像读取与检查
    left_img = cv2.imread('L.bmp')
    right_img = cv2.imread('R.jpg')
    
    assert left_img is not None, "未能读取左图L.bmp"
    assert right_img is not None, "未能读取右图R.jpg"
    
    # 左图处理流程
    print("正在处理左图...")
    left_lines = detect_laser_lines(left_img, LEFT_CONFIG)
    print(f"左图检测到{len(left_lines)}条有效激光线")
    
    # 右图处理流程
    print("\n正在处理右图...")
    right_lines = detect_laser_lines(right_img, RIGHT_CONFIG)
    print(f"右图检测到{len(right_lines)}条有效激光线")
    
    # 结果可视化
    visualize_results(left_img, left_lines, 'Left_Result')
    visualize_results(right_img, right_lines, 'Right_Result')
    
    # 保存结果
    save_clean_lines(left_lines, 'left_lines.csv')
    save_clean_lines(right_lines, 'right_lines.csv')
    print("\n结果已保存至 left_lines.csv 和 right_lines.csv")