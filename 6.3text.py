import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

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
    
    # 显著性特征参数
    'salient_n': 1,             # 亮度显著图阶数
    'entropy_thresh': 0.5,      # 最大熵分割阈值比例
    
    # 几何约束
    'roi_padding': 10,          # 边缘裁剪
    'cluster_eps': 6,           # 聚类半径
    'min_samples': 6,           # 最小样本数
    'min_line_length': 80,      # 有效线段长度

    # 中心线提取
    'window_size': 5,           # 法向计算窗口大小
    'normal_smooth': 1.5,       # 法向平滑系数
}

# ====================== 右图参数配置（针对彩色图优化） ====================== 
RIGHT_CONFIG = {
    # 基础参数
    'laser_color': 'red',       # 激光颜色类型
    'min_laser_intensity': 45,  # 最低有效激光强度

    # 预处理参数
    'clahe_clip': 2.0,          # 对比度增强上限
    'blur_kernel': (3, 3),      # 高斯模糊核大小
    'gamma_correct': 0.75,      # 高光抑制
    'specular_thresh': 180,     # 高光检测阈值

    # 显著性特征参数  
    'salient_n': 1,             # 亮度显著图阶数
    'entropy_thresh': 0.5,     # 最大熵分割阈值比例

    # 几何约束
    'roi_padding': 15,          # 边缘裁剪
    'cluster_eps': 6,           # 聚类半径
    'min_samples': 6,           # 最小样本数
    'min_line_length': 100,     # 有效线段长度

    # 中心线提取
    'window_size': 7,           # 法向计算窗口大小
    'normal_smooth': 2.0,       # 法向平滑系数
}

def lab_salient_feature(img, config):
    """
    基于LAB颜色空间的显著性特征提取
    Args:
        img: 输入BGR图像
        config: 配置参数
    Returns:
        显著性特征图
    """
    # 转换到LAB颜色空间
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 论文式(2)-(4)：重新组合颜色通道
    l_prime = l - (a + b) / 2
    a_prime = a - (l + b) / 2
    b_prime = b - (l + a) / 2
    
    # 选择L'通道作为初始显著图
    salient = np.clip(l_prime, 0, 255).astype(np.uint8)
    
    # 论文式(5)：亮度显著性增强
    avg_intensity = np.mean(salient)
    salient_map = np.zeros_like(salient, dtype=np.float32)
    
    for i in range(5, salient.shape[0]-5):
        for j in range(5, salient.shape[1]-5):
            region = salient[i-2:i+3, j-2:j+3]
            diff = abs(salient[i,j] - np.mean(region))
            salient_map[i,j] = (diff / (avg_intensity + 1e-6)) ** config['salient_n']
    
    # 归一化处理
    salient_map = cv2.normalize(salient_map, None, 0, 255, cv2.NORM_MINMAX)
    return salient_map.astype(np.uint8)

def max_entropy_segmentation(salient_map, config):
    """
    最大熵分割算法
    Args:
        salient_map: 显著性特征图
        config: 配置参数
    Returns:
        二值分割图像
    """
    # 计算灰度直方图
    hist = cv2.calcHist([salient_map], [0], None, [256], [0,256])
    hist = hist.ravel() / hist.sum()
    
    # 计算累积直方图
    cdf = hist.cumsum()
    
    # 论文式(7)-(13)：最大熵计算
    max_entropy = 0
    optimal_thresh = 0
    
    for t in range(1, 255):
        # 背景和前景的概率
        p_bg = cdf[t]
        p_fg = 1 - p_bg
        
        # 背景和前景的熵
        h_bg = 0
        for i in range(t):
            if hist[i] > 0:
                h_bg -= (hist[i]/p_bg) * np.log(hist[i]/p_bg)
                
        h_fg = 0
        for i in range(t, 256):
            if hist[i] > 0:
                h_fg -= (hist[i]/p_fg) * np.log(hist[i]/p_fg)
        
        total_entropy = h_bg + h_fg
        
        if total_entropy > max_entropy:
            max_entropy = total_entropy
            optimal_thresh = t
    
    # 应用动态阈值
    _, binary = cv2.threshold(salient_map, optimal_thresh*config['entropy_thresh'], 255, cv2.THRESH_BINARY)
    return binary

def calculate_stripe_normal(binary_img, config):
    """
    计算条纹法向场
    Args:
        binary_img: 二值分割图像
        config: 配置参数
    Returns:
        法向场矩阵 (h,w,2)
    """
    # Sobel算子计算梯度
    sobel_x = cv2.Sobel(binary_img, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(binary_img, cv2.CV_32F, 0, 1, ksize=3)
    
    # 论文式(14)-(17)：法向计算
    normal_field = np.zeros((binary_img.shape[0], binary_img.shape[1], 2), dtype=np.float32)
    
    for i in range(config['window_size'], binary_img.shape[0]-config['window_size']):
        for j in range(config['window_size'], binary_img.shape[1]-config['window_size']):
            if binary_img[i,j] > 0:
                # 计算局部窗口内的梯度统计
                window_x = sobel_x[i-config['window_size']:i+config['window_size']+1, 
                                  j-config['window_size']:j+config['window_size']+1]
                window_y = sobel_y[i-config['window_size']:i+config['window_size']+1,
                                  j-config['window_size']:j+config['window_size']+1]
                
                G_m = np.sum(window_x**2 - window_y**2)
                G_n = np.sum(2 * window_x * window_y)
                
                # 计算法向角度
                theta = 0.5 * np.arctan2(G_n, G_m)
                normal_field[i,j,0] = np.cos(theta)
                normal_field[i,j,1] = np.sin(theta)
    
    # 法向场平滑
    normal_field[:,:,0] = gaussian_filter1d(normal_field[:,:,0], config['normal_smooth'])
    normal_field[:,:,1] = gaussian_filter1d(normal_field[:,:,1], config['normal_smooth'])
    
    return normal_field

def extract_centerline(binary_img, normal_field, config):
    """
    基于法向场的灰度重心法中心线提取
    Args:
        binary_img: 二值分割图像
        normal_field: 法向场
        config: 配置参数
    Returns:
        中心线点集列表
    """
    # 论文式(18)：灰度重心法
    points = []
    
    for j in range(config['window_size'], binary_img.shape[1]-config['window_size']):
        col = binary_img[:,j]
        rows = np.where(col > 0)[0]
        
        if len(rows) > 0:
            # 在法向方向进行灰度重心计算
            center_row = int(np.mean(rows))
            normal = normal_field[center_row, j]
            
            if np.linalg.norm(normal) > 0.1:
                # 沿法向采样
                line_pts = []
                for k in range(-10, 11):
                    x = int(j + k * normal[0])
                    y = int(center_row + k * normal[1])
                    
                    if 0 <= x < binary_img.shape[1] and 0 <= y < binary_img.shape[0]:
                        line_pts.append((x, y, binary_img[y,x]))
                
                if line_pts:
                    # 计算重心
                    pts_arr = np.array(line_pts)
                    x_center = np.sum(pts_arr[:,0] * pts_arr[:,2]) / np.sum(pts_arr[:,2] + 1e-6)
                    y_center = np.sum(pts_arr[:,1] * pts_arr[:,2]) / np.sum(pts_arr[:,2] + 1e-6)
                    points.append([x_center, y_center])
    
    return np.array(points)

def detect_laser_lines(img, config):
    """激光线检测主流程（整合论文方法）"""
    # 1. 显著性特征提取
    salient_map = lab_salient_feature(img, config)
    
    # 2. 最大熵分割
    binary = max_entropy_segmentation(salient_map, config)
    
    # 3. 法向场计算
    normal_field = calculate_stripe_normal(binary, config)
    
    # 4. 中心线提取
    points = extract_centerline(binary, normal_field, config)
    
    # 5. 几何约束聚类
    if len(points) == 0:
        return []
    
    db = DBSCAN(eps=config['cluster_eps'], min_samples=config['min_samples']).fit(points)
    
    lines = []
    for label in set(db.labels_):
        if label == -1:
            continue
            
        cluster = points[db.labels_ == label]
        if len(cluster) < config['min_line_length']:
            continue
            
        # 曲线平滑
        sorted_cluster = cluster[cluster[:,1].argsort()]
        try:
            tck, u = splprep(sorted_cluster.T, s=2.0)
            new_u = np.linspace(u.min(), u.max(), int(len(u)*2))
            new_points = np.column_stack(splev(new_u, tck))
        except:
            new_points = sorted_cluster
        
        lines.append(new_points)
    
    return lines

def visualize_results(img, lines, title):
    """可视化结果"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'{title} Original')
    
    plt.subplot(122)
    vis = img.copy()
    colors = [(0,255,0), (255,0,0), (0,0,255)]
    for i, line in enumerate(lines):
        color = colors[i % len(colors)]
        pts = line.astype(int)
        cv2.polylines(vis, [pts], False, color, 2)
    
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f'Detected {len(lines)} Lines')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    left_img = cv2.imread('13.1.bmp')
    right_img = cv2.imread('13.0.bmp')

    print("处理左图(L.bmp)...")
    left_lines = detect_laser_lines(left_img, LEFT_CONFIG)
    print(f"左图提取到 {len(left_lines)} 条中心线")

    print("\n处理右图(R.jpg)...")
    right_lines = detect_laser_lines(right_img, RIGHT_CONFIG)
    print(f"右图提取到 {len(right_lines)} 条中心线")

    visualize_results(left_img, left_lines, 'Left Image')
    visualize_results(right_img, right_lines, 'Right Image')

    def save_lines(lines, filename):
        """保存中心线数据"""
        with open(filename, 'w') as f:
            for i, line in enumerate(lines):
                np.savetxt(f, line, fmt='%.2f',
                          header=f'Line {i+1}', comments='# ', delimiter=',')

    save_lines(left_lines, 'left_lines.csv')
    save_lines(right_lines, 'right_lines.csv')
    print("结果已保存为 left_lines.csv 和 right_lines.csv")