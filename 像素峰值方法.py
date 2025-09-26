import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev

# 增强配置参数
OPTIMIZED_CONFIG = {
    # 预处理参数
    'clahe_clip': 3.0,
    'blur_kernel': (5, 5),
    # 质心检测
    'dynamic_thresh_ratio': 0.4,  # 提高阈值比例
    'min_line_width': 5,         # 增加最小线宽要求
    'gap_tolerance': 7,          # 增大间隙容忍
    # 聚类参数
    'cluster_eps': 25.0,         # 增大聚类半径
    'min_samples': 10,           # 增加最小样本数
    # 后处理
    'smooth_degree': 2,
    'min_line_length': 50,       # 最小线段长度
    # 形态学处理
    'morph_kernel': np.ones((3,3), np.uint8),
    # ROI设置
    'roi_padding': 50            # 边缘裁剪
}

def optimized_preprocess(img, is_right=False):
    """增强型预处理流程"""
    # 色彩空间转换（保留彩色信息）
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img)
        # 对亮度通道进行增强
        clahe = cv2.createCLAHE(clipLimit=OPTIMIZED_CONFIG['clahe_clip'], 
                               tileGridSize=(8,8))
        l = clahe.apply(l)
        img = cv2.merge([l, a, b])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l = img[:,:,0]
        l = cv2.createCLAHE(clipLimit=OPTIMIZED_CONFIG['clahe_clip']).apply(l)
        img[:,:,0] = l
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    
    # 自适应模糊
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, OPTIMIZED_CONFIG['blur_kernel'], 0)
    
    # 针对右图的特殊处理
    if is_right:
        blurred = cv2.medianBlur(blurred, 5)
        _, blurred = cv2.threshold(blurred, 50, 255, cv2.THRESH_TOZERO)
    
    return blurred

def advanced_centroid_detection(row):
    """改进的质心检测算法"""
    # 动态阈值处理
    max_val = np.max(row)
    if max_val < 30:  # 忽略低亮度区域
        return []
    
    thresh = max_val * OPTIMIZED_CONFIG['dynamic_thresh_ratio']
    binary = np.where(row > thresh, 255, 0).astype(np.uint8)
    
    # 形态学处理填充小间隙
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                     (OPTIMIZED_CONFIG['gap_tolerance'], 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 寻找连通区域
    segments = []
    start = -1
    for i, val in enumerate(closed):
        if val == 255 and start == -1:
            start = i
        elif val == 0 and start != -1:
            if i - start >= OPTIMIZED_CONFIG['min_line_width']:
                segments.append((start, i-1))
            start = -1
    if start != -1 and len(closed) - start >= OPTIMIZED_CONFIG['min_line_width']:
        segments.append((start, len(closed)-1))
    
    # 计算加权质心
    centers = []
    for seg in segments:
        s, e = seg
        x = np.arange(s, e+1)
        weights = row[s:e+1]
        if np.sum(weights) > 0:  # 防止除以零
            centroid = int(round(np.sum(x * weights) / np.sum(weights)))
            centers.append(centroid)
    
    return centers

def refine_lines(points, img_shape):
    """线条后处理优化"""
    # 去除边缘噪声点
    h, w = img_shape
    padding = OPTIMIZED_CONFIG['roi_padding']
    mask = (points[:,0] > padding) & (points[:,0] < w - padding) & \
           (points[:,1] > padding) & (points[:,1] < h - padding)
    points = points[mask]
    
    if len(points) < OPTIMIZED_CONFIG['min_samples']:
        return []
    
    # DBSCAN聚类
    db = DBSCAN(eps=OPTIMIZED_CONFIG['cluster_eps'], 
               min_samples=OPTIMIZED_CONFIG['min_samples']).fit(points)
    
    refined_lines = []
    for label in set(db.labels_):
        if label == -1:
            continue
            
        cluster = points[db.labels_ == label]
        if len(cluster) < OPTIMIZED_CONFIG['min_line_length']:
            continue
        
        # 按y坐标排序
        sorted_cluster = cluster[cluster[:,1].argsort()]
        
        # 样条插值平滑
        try:
            # 增加插值点密度
            tck, u = splprep(sorted_cluster.T, s=OPTIMIZED_CONFIG['smooth_degree'])
            new_u = np.linspace(0, 1, int(len(sorted_cluster)*1.5))
            new_points = np.column_stack(splev(new_u, tck))
            refined_lines.append(new_points)
        except:
            # 插值失败时使用原始点
            refined_lines.append(sorted_cluster)
    
    return refined_lines

def optimized_line_detection(img, is_right=False):
    """优化的中心线提取主函数"""
    # 预处理
    processed = optimized_preprocess(img, is_right)
    
    # 逐行扫描检测质心
    all_points = []
    for y in range(processed.shape[0]):
        centers = advanced_centroid_detection(processed[y,:])
        all_points.extend([[x, y] for x in centers])
    
    if not all_points:
        return []
    
    # 转换为numpy数组
    points_array = np.array(all_points)
    
    # 线 refinement
    final_lines = refine_lines(points_array, processed.shape)
    
    # 后处理：合并相邻线段
    merged_lines = []
    if final_lines:
        # 按y坐标排序所有点
        all_points = np.concatenate(final_lines)
        all_points = all_points[all_points[:,1].argsort()]
        
        # 重新聚类合并
        merged_lines = refine_lines(all_points, processed.shape)
    
    return merged_lines if merged_lines else final_lines

def visualize_results(original, processed, lines, title):
    """增强的可视化显示"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始图像
    if len(original.shape) == 2:
        ax1.imshow(original, cmap='gray')
    else:
        ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax1.set_title(f'{title} Original')
    ax1.axis('off')
    
    # 处理后的图像
    ax2.imshow(processed, cmap='gray')
    ax2.set_title(f'{title} Processed')
    ax2.axis('off')
    
    # 检测结果
    vis = original.copy() if len(original.shape) == 3 else cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0)]
    for i, line in enumerate(lines):
        color = colors[i % len(colors)]
        pts = line.astype(int)
        for j in range(1, len(pts)):
            cv2.line(vis, tuple(pts[j-1]), tuple(pts[j]), color, 2)
    
    if len(original.shape) == 2:
        ax3.imshow(vis, cmap='gray')
    else:
        ax3.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax3.set_title(f'{title} Detected ({len(lines)} lines)')
    ax3.axis('off')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # 读取图像
    left_img = cv2.imread('L.bmp')
    right_img = cv2.imread('R.jpg')
    
    # 检查图像加载
    if left_img is None or right_img is None:
        raise ValueError("无法读取图像文件，请检查路径")
    
    # 左图处理
    print("开始处理左图...")
    left_processed = optimized_preprocess(left_img)
    left_lines = optimized_line_detection(left_img)
    print(f"左图提取到 {len(left_lines)} 条中心线")
    
    # 右图处理
    print("开始处理右图...")
    right_processed = optimized_preprocess(right_img, is_right=True)
    right_lines = optimized_line_detection(right_img, is_right=True)
    print(f"右图提取到 {len(right_lines)} 条中心线")
    
    # 可视化
    fig_left = visualize_results(left_img, left_processed, left_lines, 'Left')
    fig_right = visualize_results(right_img, right_processed, right_lines, 'Right')
    plt.show()
    
    # 保存结果
    def save_line_data(lines, filename):
        with open(filename, 'w') as f:
            for i, line in enumerate(lines):
                f.write(f"Line {i+1}:\n")
                np.savetxt(f, line, fmt='%.2f', delimiter=',')
                f.write("\n")
    
    save_line_data(left_lines, 'left_lines.txt')
    save_line_data(right_lines, 'right_lines.txt')
    print("Results saved to left_lines.txt and right_lines.txt")
