import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev

# ===== 增强版配置参数 =====
ENHANCED_CONFIG = {
    # 预处理参数
    'clahe_clip': 3.0,            # 对比度增强参数
    'blur_kernel': (7, 7),        # 自适应模糊核
    
    # 质心检测参数
    'dynamic_thresh_ratio': 0.25, # 动态阈值系数
    'min_line_width': 3,          # 最小有效条纹宽度
    'gap_tolerance': 5,           # 横向间断容忍度
    
    # 聚类参数
    'cluster_eps': 20.0,          # 增大聚类半径
    'min_samples': 8,             # 减少最小样本数
    
    # 后处理参数
    'smooth_degree': 3,           # 样条平滑度
    'interp_step': 0.5            # 插值步长
}

def enhance_contrast(img):
    """对比度增强处理"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=ENHANCED_CONFIG['clahe_clip'], 
                           tileGridSize=(8,8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)

def adaptive_blur(img):
    """自适应高斯模糊"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    return cv2.GaussianBlur(gray, ENHANCED_CONFIG['blur_kernel'], 0)

def robust_centroid_detection(row):
    """
    抗干扰质心检测
    
    参数:
        row (numpy.ndarray): 输入的一维数组，图像行的像素强度值
    
    返回值:
        list[int]: 检测到的质心x坐标列表，从左到右
    
    实现说明:
        通过动态阈值和形态学处理增强信号可靠性，采用加权平均法计算质心坐标
    """
    # 动态阈值处理（基于行最大值动态调整阈值）
    thresh = np.max(row) * ENHANCED_CONFIG['dynamic_thresh_ratio']
    binary = cv2.threshold(row, thresh, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
    
    # 形态学闭运算（填充横向间隙）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ENHANCED_CONFIG['gap_tolerance'],1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 连通区域检测（识别有效信号区间）
    segments = []
    start = -1
    for i, val in enumerate(closed):
        if val and start == -1:
            start = i
        elif not val and start != -1:
            segments.append((start, i))
            start = -1
    if start != -1:
        segments.append((start, len(closed)))
    
    # 加权质心计算（排除过小信号段）
    centers = []
    for s, e in segments:
        if e - s >= ENHANCED_CONFIG['min_line_width']:
            x = np.arange(s, e)
            weights = row[s:e]
            centroid = np.sum(x * weights) / np.sum(weights)
            centers.append(int(centroid))
    return centers

def interpolate_points(points):
    """
    使用样条插值算法填补间断的点序列
    
        points (list/np.ndarray): 二维点坐标序列，形状应为(N,2)的数组
        
    返回值:
        np.ndarray: 经过插值处理后的新点坐标序列，形状为(M,2)的数组
        
        当插值失败时返回原始输入点序列
    """
    
    # 点数量不足时直接返回（样条插值需要至少4个控制点）
    if len(points) < 4: 
        return points
    
    # 转换为numpy数组以便进行矩阵操作
    points = np.array(points)
    
    try:
        # 计算样条曲线参数（s控制平滑度，从配置获取参数）
        tck, _ = splprep(points.T, s=ENHANCED_CONFIG['smooth_degree'])
        
        # 生成插值参数序列（步长根据原始点数动态调整）
        u = np.arange(0, 1, ENHANCED_CONFIG['interp_step']/len(points))
        
        # 执行样条插值计算并重组为坐标点矩阵
        return np.column_stack(splev(u, tck))
    except:
        # 异常时返回原始数据（保持数据完整性）
        return points

def detect_connected_lines(img):
    """
    增强版多线检测函数，通过图像处理和多阶段分析识别图像中的连续线条
    
        img: numpy.ndarray 输入图像矩阵，应为二维灰度图像数组
    
    Returns:
        list: 检测到的线条列表，每个元素为经过插值处理的点数组，表示一条连续线条
    """
    
    # 预处理流程：提升对比度与自适应降噪
    # 通过对比度增强突出线条特征，采用自适应模糊减少噪声干扰
    enhanced = enhance_contrast(img)
    blurred = adaptive_blur(enhanced)
    
    # 逐行中心点检测：垂直扫描获取候选点
    # 对图像每一行进行鲁棒的中心点检测，收集所有候选坐标
    points = []
    for y in range(blurred.shape[0]):
        row = blurred[y, :]
        centers = robust_centroid_detection(row)
        points.extend([[x, y] for x in centers])
    
    # 密度聚类分析：过滤噪声并分组候选点
    # 当有效点过少时直接返回空结果，使用改进参数的DBSCAN进行空间聚类
    if len(points) < 10:
        return []
    
    db = DBSCAN(eps=ENHANCED_CONFIG['cluster_eps'], 
               min_samples=ENHANCED_CONFIG['min_samples']).fit(points)
    
    # 后处理流程：生成连续线条
    # 对每个聚类点集进行纵向排序，并通过插值得到平滑的线条结构
    lines = []
    for label in set(db.labels_):
        if label == -1:  # 跳过噪声点簇
            continue
        cluster = np.array(points)[db.labels_ == label]
        sorted_cluster = cluster[cluster[:,1].argsort()]  # 按y坐标升序排列
        interpolated = interpolate_points(sorted_cluster)  # 样条插值处理
        lines.append(interpolated)
    
    return lines

def save_coordinates(lines, filename):
    """保存中心线坐标矩阵
        lines: 中心线坐标列表，每个元素为二维numpy数组，表示一条中心线的坐标点集合
        filename: 输出文件名，坐标数据将写入此文本文件
        无返回值，结果直接写入文件
    """
    with open(filename, 'w') as f:
        for line_idx, line in enumerate(lines):
            # 处理单条中心线坐标点：
            # 1. 按y轴坐标升序排列坐标点
            # 2. 过滤x/y坐标非正数的无效点
            sorted_line = line[line[:,1].argsort()]
            valid_points = sorted_line[(sorted_line[:,0] > 0) & (sorted_line[:,1] > 0)]
            
            # 写入当前中心线头信息（线编号和有效点数）
            f.write(f"# Line {line_idx+1} (Points: {len(valid_points)})\n")
            
            # 将有效坐标点以整数格式写入文件，逗号分隔
            np.savetxt(f, valid_points, fmt='%d', delimiter=',')
            f.write("\n")

def visualize_lines(img, lines, thickness=2):
    """可视化函数）"""
    vis = img.copy() if len(img.shape)==3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    colors = [(0,255,0), (0,0,255), (255,0,255), (255,255,0)]
    
    for i, line in enumerate(lines):
        color = colors[i % len(colors)]
        # 插值点转换为整数坐标
        int_line = line.astype(int)
        for j in range(1, len(int_line)):
            cv2.line(vis, tuple(int_line[j-1]), tuple(int_line[j]), color, thickness)
    return vis

# ===== 完整主程序 =====
if __name__ == "__main__":
    try:
        left_img = cv2.imread('L.bmp')
        right_img = cv2.imread('R.jpg')
        assert left_img is not None and right_img is not None
    except:
        raise FileNotFoundError("请确保L.jpg和R.jpg存在于当前目录且可读取")

    # 处理流程（添加进度显示）
    print("正在处理左图...")
    left_lines = detect_connected_lines(left_img)
    print("正在处理右图...")
    right_lines = detect_connected_lines(right_img)

    # 保存结果
    save_coordinates(left_lines, 'L.txt')
    save_coordinates(right_lines, 'R.txt')
    print(f"坐标文件已保存：L.txt ({len(left_lines)}条线), R.txt ({len(right_lines)}条线)")

    fig = plt.figure(figsize=(16, 8))
    
    # 左图对比
    ax1 = fig.add_subplot(231)
    ax1.imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Left Original')
    
    ax2 = fig.add_subplot(232)
    ax2.imshow(cv2.cvtColor(visualize_lines(left_img, left_lines), cv2.COLOR_BGR2RGB))
    ax2.set_title('Left Processed')

    # 右图对比
    ax3 = fig.add_subplot(234)
    ax3.imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
    ax3.set_title('Right Original')
    
    ax4 = fig.add_subplot(235)
    ax4.imshow(cv2.cvtColor(visualize_lines(right_img, right_lines), cv2.COLOR_BGR2RGB))
    ax4.set_title('Right Processed')

    plt.tight_layout()
    plt.show()