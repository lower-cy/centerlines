import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev

ENHANCED_CONFIG = {
    # 全局参数
    'laser_color': 'red',  # 激光颜色类型
    'min_laser_intensity': 70,  #30-100	激光强度阈值：值越大，过滤弱光点越多
    
    # 预处理参数
    'clahe_clip': 4.0,      #1.0-5.0	对比度增强强度：值越大，局部对比度增强越强，但可能引入噪点
    'blur_kernel': (3, 3),  #奇数3x3~9x9	平滑程度：核越大，模糊效果越强，细节丢失越多
    'morph_kernel': (5,5),  # 形态学核尺寸
    
    # 区域分割参数
    'roi_padding': 35,      #0-100	边缘裁剪宽度：值越大，保留中心区域越多
    'dynamic_roi': True,    # 是否启用动态ROI
    
    # 质心检测参数
    'dynamic_thresh_ratio': 0.4,  #0.2-0.6	动态阈值比例：值越大，检测点越少，抗噪性越强
    'min_line_width': 5,          #3-10	最小线宽：值越大，过滤短线段越严格
    'max_line_gap': 7,            #3-15	最大间隙容忍：值越大，允许的横向断裂越长
    
    # 聚类参数
    'cluster_eps': 25.0,          #10-50	聚类半径：值越大，合并的点越多
    'min_samples': 15,            #5-20	最小样本数：值越大，聚类越严格
    'min_line_length': 90,        #30-100	有效线段长度：值越大，保留线段越长
    
    # 后处理参数
    'smooth_degree': 2,            #1-3	平滑强度：值越大，曲线越平滑但可能失真
    'interp_step': 0.2,            #0.1-0.5	平滑步长：值越大，曲线越平滑但可能失真
    
    # 抗高光参数
    'specular_thresh': 240,  # 高光区域阈值    150-250	高光检测阈值：值越小，更多区域被视为高光
    'gamma_correct': 0.3     # 伽马校正系数    0.3-1.0	高光抑制强度：值越小，高光区域压暗越明显
}

def enhance_laser_channel(img):
    """自适应激光通道增强"""
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    b, g, r = cv2.split(img)
    if ENHANCED_CONFIG['laser_color'] == 'red':
        enhanced = cv2.addWeighted(r, 2.0, cv2.add(b, g), -0.8, 0)
    elif ENHANCED_CONFIG['laser_color'] == 'green':
        enhanced = cv2.addWeighted(g, 2.0, cv2.add(r, b), -0.8, 0)
    else: # blue
        enhanced = cv2.addWeighted(b, 2.0, cv2.add(r, g), -0.8, 0)
    
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.merge([enhanced, enhanced, enhanced])

def adaptive_gamma_correction(img):
    """自适应伽马校正抑制高光"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 检测高光区域
    _, specular_mask = cv2.threshold(gray, ENHANCED_CONFIG['specular_thresh'], 255, cv2.THRESH_BINARY)
    
    # 伽马校正
    inv_gamma = 1.0 / ENHANCED_CONFIG['gamma_correct']
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    # 仅在高光区域应用校正
    corrected = cv2.LUT(img, table)
    return cv2.bitwise_and(corrected, corrected, mask=specular_mask) + \
           cv2.bitwise_and(img, img, mask=~specular_mask)

def multi_scale_preprocess(img):
    """多尺度预处理流水线"""
    # 伽马校正抑制高光
    img = adaptive_gamma_correction(img)
    
    # 转换到LAB空间处理亮度
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 自适应CLAHE
    clahe = cv2.createCLAHE(clipLimit=ENHANCED_CONFIG['clahe_clip'], 
                           tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # 多尺度模糊
    blur1 = cv2.GaussianBlur(l, (5,5), 0)
    blur2 = cv2.medianBlur(l, 5)
    merged = cv2.addWeighted(blur1, 0.7, blur2, 0.3, 0)
    
    # 增强激光通道
    merged = enhance_laser_channel(merged)
    return cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)

def dynamic_centroid_detection(row):
    """动态阈值质心检测"""
    # 排除低亮度行
    max_val = np.max(row)
    if max_val < ENHANCED_CONFIG['min_laser_intensity']:
        return []
    
    # 动态阈值计算
    thresh = max_val * ENHANCED_CONFIG['dynamic_thresh_ratio']
    binary = np.where(row > thresh, 255, 0).astype(np.uint8)
    
    # 横向连接处理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                     (ENHANCED_CONFIG['max_line_gap'], 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 寻找有效线段
    segments = []
    start = -1
    for i, val in enumerate(closed):
        if val == 255 and start == -1:
            start = i
        elif val == 0 and start != -1:
            if i - start >= ENHANCED_CONFIG['min_line_width']:
                segments.append((start, i-1))
            start = -1
    if start != -1 and len(closed)-start >= ENHANCED_CONFIG['min_line_width']:
        segments.append((start, len(closed)-1))
    
    # 亚像素质心计算
    centers = []
    for s, e in segments:
        x = np.arange(s, e+1)
        weights = row[s:e+1]
        if np.sum(weights) == 0:
            continue
        centroid = np.sum(x * weights) / np.sum(weights)
        centers.append(int(round(centroid)))
    
    return centers

def geometry_based_clustering(points, img_size):
    """基于几何约束的聚类优化"""
    # 去除边缘区域点
    h, w = img_size
    mask = (points[:,0] > ENHANCED_CONFIG['roi_padding']) & \
           (points[:,0] < w - ENHANCED_CONFIG['roi_padding']) & \
           (points[:,1] > ENHANCED_CONFIG['roi_padding']) & \
           (points[:,1] < h - ENHANCED_CONFIG['roi_padding'])
    points = points[mask]
    
    if len(points) < ENHANCED_CONFIG['min_samples']:
        return []
    
    # DBSCAN聚类
    db = DBSCAN(eps=ENHANCED_CONFIG['cluster_eps'], 
               min_samples=ENHANCED_CONFIG['min_samples']).fit(points)
    
    valid_lines = []
    for label in set(db.labels_):
        if label == -1:
            continue
            
        cluster = points[db.labels_ == label]
        if len(cluster) < ENHANCED_CONFIG['min_line_length']:
            continue
        
        # 按y排序并插值
        sorted_cluster = cluster[cluster[:,1].argsort()]
        
        # 分段线性插值
        try:
            tck, u = splprep(sorted_cluster.T, s=ENHANCED_CONFIG['smooth_degree'])
            new_u = np.linspace(u.min(), u.max(), int(len(u)*2))
            new_points = np.column_stack(splev(new_u, tck))
            valid_lines.append(new_points)
        except:
            valid_lines.append(sorted_cluster)
    
    # 方向一致性过滤
    final_lines = []
    for line in valid_lines:
        dy = line[-1,1] - line[0,1]
        dx = line[-1,0] - line[0,0]
        if dy == 0:
            continue
        angle = np.arctan(dx/dy)
        if abs(angle) > 0.5:  # 过滤角度过大线段
            continue
        final_lines.append(line)
    
    return final_lines

def detect_laser_lines(img):
    """激光线检测主流程"""
    # 预处理
    preprocessed = multi_scale_preprocess(img)
    
    # 逐行质心检测
    points = []
    for y in range(preprocessed.shape[0]):
        centers = dynamic_centroid_detection(preprocessed[y,:])
        points.extend([[x, y] for x in centers])
    
    if not points:
        return []
    
    # 几何约束聚类
    lines = geometry_based_clustering(np.array(points), preprocessed.shape)
    return lines

def visualize_analysis(img, lines):
    """增强可视化分析"""
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    
    # 原始图像
    ax[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0,0].set_title('Original Image')
    
    # 预处理效果
    preprocessed = multi_scale_preprocess(img)
    ax[0,1].imshow(preprocessed, cmap='gray')
    ax[0,1].set_title('Preprocessed Result')
    
    # 质心分布
    points = np.concatenate(lines) if lines else []
    ax[1,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if len(points) > 0:
        ax[1,0].scatter(points[:,0], points[:,1], c='r', s=5)
    ax[1,0].set_title('Centroid Distribution')
    
    # 最终结果
    vis = img.copy()
    colors = [(0,255,0), (255,0,0), (0,0,255)]
    for i, line in enumerate(lines):
        color = colors[i % len(colors)]
        pts = line.astype(int)
        cv2.polylines(vis, [pts], False, color, 2)
    ax[1,1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax[1,1].set_title(f'Detected {len(lines)} Lines')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 示例用法
    left_img = cv2.imread('L.bmp')
    right_img = cv2.imread('R.jpg')
    
    print("提取L图中心线...")
    left_lines = detect_laser_lines(left_img)
    print(f"提取到L图 {len(left_lines)} 中心线")
    
    print("提取R图中心线...")
    right_lines = detect_laser_lines(right_img)
    print(f"提取到R图 {len(right_lines)} 中心线")
    
    # 可视化分析
    visualize_analysis(left_img, left_lines)
    visualize_analysis(right_img, right_lines)
    
    # 保存结果
    def save_results(lines, filename):
        with open(filename, 'w') as f:
            for i, line in enumerate(lines):
                np.savetxt(f, line, fmt='%.2f', 
                          header=f'Line {i+1}', 
                          comments='# ', 
                          delimiter=',')
    save_results(left_lines, 'left_lines.csv')
    save_results(right_lines, 'right_lines.csv')