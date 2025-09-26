import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d

# ====================== 参数配置 ======================
"""
参数设计原则：
1. 左图（L.bmp）为灰度图像，右图（R.jpg）为彩色图像
2. 核心参数分类：
   - 预处理参数：影响图像增强和噪声抑制
   - 形态学参数：控制缺口填充和形状修正
   - 检测参数：决定中心线提取灵敏度
   - 后处理参数：影响曲线平滑和异常点过滤
"""

# ----------------- 左图参数 -----------------
LEFT_CONFIG = {
    # 预处理参数
    'gamma': 0.5,                 # 伽马校正[0.3-1.0]，值越小增强暗部越强
    'clahe_clip': 3.0,            # CLAHE对比度限制[1.0-5.0]，值越大增强越强
    'blur_size': (7,7),           # 高斯模糊核大小[(3,3)-(9,9)]，奇数，大尺寸抑制噪声更好
    'local_enhance_region': 0.7,  # 右侧增强起始位置比例[0.5-0.9]，值越小增强区域越大
    
    # 形态学参数
    'morph_kernel': (9,9),        # 形态学操作核尺寸[(5,5)-(15,15)]，大尺寸填补更大缺口
    'morph_iterations': 3,        # 形态学操作迭代次数[1-5]，值大填补效果强但可能过度
    
    # 检测参数
    'min_intensity': 30,          # 激光最小强度[10-100]，值大过滤更多弱信号
    'peak_window': 15,            # 峰值检测窗口大小[5-25]，奇数，值小检测更精细
    'max_gap': 50,                # 最大横向间断距离[20-100]，值大允许更长间断
    
    # 后处理参数
    'cluster_eps': 15.0,          # 聚类半径[10.0-30.0]，值小生成更多小段
    'min_samples': 8,             # 最小聚类点数[5-20]，值大过滤更多孤立点
    'smooth_sigma': 3.0           # 高斯平滑强度[1.0-5.0]，值大曲线更平滑
}

# ----------------- 右图参数 ----------------- 
RIGHT_CONFIG = {
    # 预处理参数
    'gamma': 0.7,                 
    'clahe_clip': 4.0,
    'blur_size': (7,7),
    'local_enhance_region': 0.7,
    
    # 形态学参数  
    'morph_kernel': (9,9),       # 更大核处理更多缺口
    'morph_iterations': 3,
    
    # 检测参数
    'min_intensity': 40,           # 更高阈值抑制图案干扰
    'peak_window': 21,             # 更大窗口适应更宽条纹
    'max_gap': 80,
    
    # 后处理参数
    'cluster_eps': 20.0,
    'min_samples': 10,
    'smooth_sigma': 4.0
}

# ====================== 核心算法 ======================
def adaptive_preprocess(img, config, is_left=True):
    """
    自适应预处理流程
    流程：伽马校正 → 局部CLAHE → 高斯模糊 → 虚影抑制
    """
    # 伽马校正增强暗部
    gamma_table = np.array([(i/255.0)**(1.0/config['gamma'])*255 for i in range(256)]).astype('uint8')
    gamma_img = cv2.LUT(img, gamma_table)
    
    # 转换色彩空间
    if is_left:
        gray = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2GRAY)  # 灰度转换
    else:
        hsv = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2HSV)
        gray = hsv[:,:,2]  # 使用Value通道

    # 局部CLAHE增强右侧
    clahe = cv2.createCLAHE(clipLimit=config['clahe_clip'], tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 混合增强：仅增强右侧区域
    h, w = gray.shape
    mask = np.zeros_like(gray, dtype=np.uint8)  # 创建掩膜
    start_col = int(w * config['local_enhance_region'])
    mask[:, start_col:] = 255  # 将需要增强的区域设置为白色（255）
    
    # 分步实现带掩膜的混合
    blended_part = cv2.addWeighted(gray, 0.3, enhanced, 0.7, 0)  # 混合图像
    blended = cv2.bitwise_or(
        cv2.bitwise_and(blended_part, blended_part, mask=mask),  # 提取增强部分
        cv2.bitwise_and(gray, gray, mask=cv2.bitwise_not(mask))  # 提取保留原图部分
    )
    
    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(blended, config['blur_size'], 0)
    
    return blurred

def morphological_repair(img, config):
    """
    形态学修复缺口
    流程：闭运算填补缺口 → 顶帽增强对比
    """
    # 闭运算填补缺口
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config['morph_kernel'])
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=config['morph_iterations'])
    
    # 顶帽操作增强对比
    tophat = cv2.morphologyEx(closed, cv2.MORPH_TOPHAT, (5,5))
    
    return cv2.add(closed, tophat)

def detect_peak_centroids(row, config):
    """
    改进的质心检测算法
    特点：动态基线 + 滑动窗口峰值检测
    """
    # 平滑处理
    smoothed = cv2.GaussianBlur(row, (config['peak_window'], 1), 0)
    
    # 动态基线计算（排除高光区）
    avg_baseline = np.mean(smoothed[smoothed < 100])
    dynamic_thresh = avg_baseline + 2 * np.std(smoothed)
    
    # 滑动窗口检测局部峰值
    centroids = []
    window_half = config['peak_window'] // 2
    for i in range(window_half, len(smoothed) - window_half):
        window = smoothed[i - window_half:i + window_half + 1]
        if np.argmax(window) == window_half and smoothed[i] > dynamic_thresh:
            # 亚像素级精确定位
            x = [i - 1, i, i + 1]
            y = [smoothed[i - 1], smoothed[i], smoothed[i + 1]]
            coeff = np.polyfit(x, y, 2)
            peak_pos = -coeff[1] / (2 * coeff[0])
            centroids.append(peak_pos)  # 直接添加标量值
    
    return np.array(centroids).flatten()  # 确保返回一维数组

def cluster_and_interpolate(points, config, img_shape):
    """
    聚类与插值处理
    流程：DBSCAN聚类 → 样条插值 → 高斯平滑
    """
    # 排除边缘区域
    h, w = img_shape
    valid_mask = (points[:,0] > w*0.1) & (points[:,0] < w*0.9) & (points[:,1] > h*0.1)
    filtered = points[valid_mask]
    
    # DBSCAN聚类
    db = DBSCAN(eps=config['cluster_eps'], min_samples=config['min_samples']).fit(filtered)
    
    lines = []
    for label in np.unique(db.labels_):
        if label == -1:
            continue
        cluster = filtered[db.labels_ == label]
        if len(cluster) < 50:  # 过滤小聚类
            continue
        
        # 按y坐标排序
        sorted_cluster = cluster[cluster[:,1].argsort()]
        
        # 样条插值
        try:
            tck, u = splprep(sorted_cluster.T, s=config['smooth_sigma']*len(cluster), k=3)
            new_points = np.column_stack(splev(np.linspace(0,1,200), tck))
            lines.append(new_points)
        except:
            lines.append(sorted_cluster)
    
    return lines

def process_image(img, config, is_left=True):
    """
    完整处理流程
    """
    # 预处理
    preprocessed = adaptive_preprocess(img, config, is_left)
    
    # 形态学修复
    repaired = morphological_repair(preprocessed, config)
    
    # 逐行检测质心
    h, w = repaired.shape
    points = []
    for y in range(h):
        row = repaired[y, :]
        centroids = detect_peak_centroids(row, config)
        if len(centroids) > 0:  # 确保有检测到的质心
            points.extend([[x, y] for x in centroids])
    
    if not points:
        return []
    
    # 验证 points 的形状
    try:
        points_array = np.array(points)
        print("Points array shape:", points_array.shape)
    except ValueError as e:
        print("Error converting points to array:", e)
        print("Points content:", points[:10])  # 打印前10个点
        raise
    
    # 聚类与插值
    lines = cluster_and_interpolate(points_array, config, (h, w))
    
    return lines

# ====================== 可视化与输出 ======================
def visualize_processing(img, lines, title):
    """分步可视化处理过程"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 原始图像
    axes[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title(f'{title} Original')
    
    # 预处理效果
    preprocessed = adaptive_preprocess(img, LEFT_CONFIG if 'Left' in title else RIGHT_CONFIG, 'Left' in title)
    axes[0,1].imshow(preprocessed, cmap='gray')
    axes[0,1].set_title('Preprocessed')
    
    # 形态学修复结果
    repaired = morphological_repair(preprocessed, LEFT_CONFIG if 'Left' in title else RIGHT_CONFIG)
    axes[0,2].imshow(repaired, cmap='gray')
    axes[0,2].set_title('Morphological Repair')
    
    # 质心检测点
    axes[1,0].scatter([p[0] for line in lines for p in line], 
                     [p[1] for line in lines for p in line], s=1)
    axes[1,0].set_title('Raw Centroids')
    axes[1,0].invert_yaxis()
    
    # 最终结果
    vis = img.copy()
    for line in lines:
        pts = line.astype(int)
        cv2.polylines(vis, [pts], False, (0,255,0), 2)
    axes[1,1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[1,1].set_title('Final Result')
    
    # 3D预览
    ax3d = fig.add_subplot(236, projection='3d')
    for line in lines:
        ax3d.plot(line[:,0], line[:,1], np.zeros_like(line[:,0]))
    ax3d.set_title('3D Preview')
    
    plt.tight_layout()
    plt.show()

def save_lines(lines, filename):
    """保存为CSV文件"""
    with open(filename, 'w') as f:
        for i, line in enumerate(lines):
            np.savetxt(f, line, fmt='%.2f', 
                      header=f'Line_{i+1}', 
                      comments='# ',
                      delimiter=',')
            f.write('\n\n')

# ====================== 主程序 ======================
if __name__ == "__main__":
    # 读取图像
    left_img = cv2.imread('L.bmp')
    right_img = cv2.imread('R.jpg')
    
    assert left_img is not None, "未能读取左图L.bmp"
    assert right_img is not None, "未能读取右图R.jpg"
    
    # 处理左图
    print("处理左图中...")
    left_lines = process_image(left_img, LEFT_CONFIG, is_left=True)
    print(f"检测到 {len(left_lines)} 条左激光线")
    
    # 处理右图
    print("\n处理右图中...")
    right_lines = process_image(right_img, RIGHT_CONFIG, is_left=False)
    print(f"检测到 {len(right_lines)} 条右激光线")
    
    # 可视化
    visualize_processing(left_img, left_lines, 'Left Processing')
    visualize_processing(right_img, right_lines, 'Right Processing')
    
    # 保存结果
    save_lines(left_lines, 'left_lines.csv')
    save_lines(right_lines, 'right_lines.csv')
    print("\n结果已保存")