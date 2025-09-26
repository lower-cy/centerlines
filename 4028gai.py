import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev
from collections import defaultdict

# ================== 配置参数 ==================
CONFIG = {
    # 相机标定参数
    'camera_matrix_left': np.array([
        [4703.3840666469305, 0.0, 1133.8966264844476],
        [0.0, 4657.770006641158, 983.7755276735744],
        [0.0, 0.0, 1.0]]),
    'dist_coeffs_left': np.array([
        -0.19060368249367288, -6.827044122904246, 
        0.015377030028687984, -0.00750634791176898, 
        107.39588017569562]),
    'camera_matrix_right': np.array([
        [4409.199175099535, 0.0, 1531.0013908252736],
        [0.0, 4384.905205883512, 1013.4751888939345],
        [0.0, 0.0, 1.0]]),
    'dist_coeffs_right': np.array([
        -0.42270673798875497, 1.378263372731151, 
        0.009909410979026863, -0.008593483642757997, 
        -1.0961258361436514]),
    'R': np.array([[0.9867230542685737, 0.007483211056180142, 0.1622393778562597],
                [-0.005753664364150946, 0.9999215317777955, -0.011127696685821956],
                [-0.16230991812357692, 0.010046483933974946, 0.9866886837494805]]),  # 旋转矩阵
    'T': np.array([[-65.930698300496], [0.7317230319931822], [-12.020455702540955]]),  # 平移向量
    
    # 图像处理参数
    'clahe_clip': 2.0,
    'blur_kernel': (5, 5),
    'mask_ratio': 0.4,
    
    # 中心线检测
    'dynamic_thresh': 0.3,
    'min_line_width': 5,
    'gap_tolerance': 7,
    
    # 匹配参数
    'max_disparity': 150,
    'min_disparity': 20,
    'ncc_threshold': 0.7,
    'match_window': 15
}

# ================== 核心函数 ==================

def stereo_rectify(left, right):
    """
    立体校正
    该函数通过立体校正过程，将左右相机的图像转换为扫描线对齐的图像
    这有助于后续的立体匹配和深度图计算
    :param left: 左相机图像
    :param right: 右相机图像
    :return: 校正后的左右图像
    """
    # 执行立体校正计算，获取校正所需的旋转矩阵、投影矩阵和重投影矩阵等
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        CONFIG['camera_matrix_left'], CONFIG['dist_coeffs_left'],
        CONFIG['camera_matrix_right'], CONFIG['dist_coeffs_right'],
        left.shape[:2], CONFIG['R'], CONFIG['T'],
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9
    )
    
    # 生成校正映射，用于将图像从原始坐标系转换到校正后的坐标系
    map1_l, map2_l = cv2.initUndistortRectifyMap(
        CONFIG['camera_matrix_left'], CONFIG['dist_coeffs_left'], R1, P1,
        left.shape[:2], cv2.CV_16SC2
    )
    map1_r, map2_r = cv2.initUndistortRectifyMap(
        CONFIG['camera_matrix_right'], CONFIG['dist_coeffs_right'], R2, P2,
        right.shape[:2], cv2.CV_16SC2
    )
    
    # 应用校正映射，重映射原始图像到校正后的坐标系中
    rect_left = cv2.remap(left, map1_l, map2_l, cv2.INTER_LINEAR)
    rect_right = cv2.remap(right, map1_r, map2_r, cv2.INTER_LINEAR)
    
    # 返回校正后的左右图像
    return rect_left, rect_right

def enhance_contrast(img):
    """对比度增强

    使用CLAHE（限制对比度的自适应直方图均衡化）方法增强图像的对比度。
    CLAHE是一种高级的图像处理技术，它将图像分成小区域并进行直方图均衡化，
    从而提高图像的局部对比度，同时避免了全局直方图均衡化可能引入的噪声问题。

    参数:
    img: 输入的BGR图像。类型为numpy数组。

    返回值:
    增强对比度后的BGR图像。类型为numpy数组。
    """
    # 将图像从BGR颜色空间转换到LAB颜色空间
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # 分离LAB图像的三个通道
    l, a, b = cv2.split(lab)
    
    # 创建CLAHE对象，配置CLAHE的参数
    clahe = cv2.createCLAHE(clipLimit=CONFIG['clahe_clip'], tileGridSize=(8,8))
    
    # 合并处理后的L通道和原始的A、B通道，并将颜色空间转换回BGR
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

def extract_centroid(row):
    """质心检测"""
    thresh = np.max(row) * CONFIG['dynamic_thresh']
    binary = cv2.threshold(row, thresh, 255, cv2.THRESH_BINARY)[1]
    
    # 连通域处理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (CONFIG['gap_tolerance'],1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 分段计算
    segments = []
    start = -1
    for i, val in enumerate(closed):
        if val > 0 and start == -1:
            start = i
        elif val == 0 and start != -1:
            if (i - start) >= CONFIG['min_line_width']:
                segments.append((start, i))
            start = -1
    if start != -1:
        segments.append((start, len(closed)))
    
    # 计算质心
    centers = []
    for s, e in segments:
        x = np.arange(s, e)
        weights = row[s:e]
        centroid = np.sum(x * weights) / (np.sum(weights) + 1e-6)
        centers.append(int(centroid))
    return centers

def cluster_lines(points):
    """线特征聚类"""
    if len(points) < 10:
        return []
    
    db = DBSCAN(eps=20, min_samples=5).fit(points)
    lines = []
    for label in set(db.labels_):
        if label == -1:
            continue
        cluster = points[db.labels_ == label]
        sorted_cluster = cluster[cluster[:,1].argsort()]
        
        # 样条插值
        try:
            tck, _ = splprep(sorted_cluster.T, s=2)
            u = np.linspace(0, 1, 100)
            interp = np.column_stack(splev(u, tck))
            lines.append(interp.astype(int))
        except:
            lines.append(sorted_cluster)
    return lines

def stereo_match(left_lines, right_lines, gray_L, gray_R):
    """立体匹配"""
    h, w = gray_L.shape
    half_win = CONFIG['match_window'] // 2
    matches = []
    right_dict = defaultdict(list)
    
    # 构建右图索引
    for idx, line in enumerate(right_lines):
        for x, y in line:
            right_dict[int(y)].append((x, idx))
    
    used_right = set()
    
    # 遍历左图线条
    for l_idx, l_line in enumerate(left_lines):
        best_score = -1
        best_r_idx = None
        
        # 提取有效点
        valid_points = [p for p in l_line if 
                       (half_win <= p[0] < w-half_win) and 
                       (half_win <= p[1] < h-half_win)]
        if not valid_points:
            continue
        
        # 动态视差范围
        lx_avg = np.mean([p[0] for p in valid_points])
        min_d = max(CONFIG['min_disparity'], int(lx_avg) - CONFIG['max_disparity'])
        max_d = int(lx_avg) - CONFIG['min_disparity']
        
        # 候选匹配收集
        candidates = defaultdict(list)
        for (x, y) in valid_points:
            for y_off in range(-3, 4):
                ry = y + y_off
                for rx, r_idx in right_dict.get(ry, []):
                    if min_d <= (x - rx) <= max_d:
                        candidates[r_idx].append((x, y, rx, ry))
        
        # 评估候选
        for r_idx, pairs in candidates.items():
            if r_idx in used_right:
                continue
            
            scores = []
            for lx, ly, rx, ry in pairs:
                patch_L = gray_L[ly-half_win:ly+half_win+1, lx-half_win:lx+half_win+1]
                patch_R = gray_R[ry-half_win:ry+half_win+1, rx-half_win:rx+half_win+1]
                
                if patch_L.shape == patch_R.shape:
                    ncc = np.corrcoef(patch_L.flatten(), patch_R.flatten())[0,1]
                    scores.append(ncc)
            
            if scores:
                avg_score = np.mean(scores)
                if avg_score > best_score and avg_score > CONFIG['ncc_threshold']:
                    best_score = avg_score
                    best_r_idx = r_idx
        
        if best_r_idx is not None:
            matches.append((l_line, right_lines[best_r_idx]))
            used_right.add(best_r_idx)
    
    return matches

def visualize_results(rect_left, rect_right, matches):
    """可视化结果"""
    # 创建合成图像
    composite = np.hstack([rect_left, rect_right])
    composite = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
    offset = rect_left.shape[1]
    
    # 绘制中心线
    colormap = plt.cm.get_cmap('hsv', len(matches))
    for idx, (l_line, r_line) in enumerate(matches):
        color = np.array(colormap(idx)[:3]) * 255  # 转换为0-255范围
        color = color.astype(np.uint8).tolist()
        
        # 绘制左图线条
        for i in range(1, len(l_line)):
            cv2.line(composite, 
                     tuple(l_line[i-1].astype(int)), 
                     tuple(l_line[i].astype(int)), 
                     color, 2)
        
        # 绘制右图线条（偏移到右侧）
        r_line_offset = r_line + np.array([offset, 0])
        for i in range(1, len(r_line)):
            cv2.line(composite,
                     tuple(r_line_offset[i-1].astype(int)),
                     tuple(r_line_offset[i].astype(int)),
                     color, 2)
        
        # 绘制虚线连接对应点
        for i in range(len(l_line)):
            l_point = l_line[i].astype(int)
            r_point = r_line_offset[i].astype(int)
            cv2.line(composite, 
                     tuple(l_point), 
                     tuple(r_point), 
                     color, 1, lineType=cv2.LINE_AA)
    
    plt.figure(figsize=(20,10))
    plt.imshow(composite)
    plt.axis('off')
    plt.title("Stereo Matching Results with Epipolar Lines")
    plt.show()

# ================== 主流程 ==================
if __name__ == "__main__":
    # 加载原始图像
    raw_left = cv2.imread('L.bmp')
    raw_right = cv2.imread('R.jpg')
    assert raw_left is not None and raw_right is not None, "图像加载失败"
    
    # 立体校正
    rect_left, rect_right = stereo_rectify(raw_left, raw_right)
    
    # 预处理
    proc_left = enhance_contrast(rect_left)
    proc_left = cv2.GaussianBlur(proc_left, CONFIG['blur_kernel'], 0)
    gray_left = cv2.cvtColor(proc_left, cv2.COLOR_BGR2GRAY)
    
    proc_right = enhance_contrast(rect_right)
    proc_right = cv2.GaussianBlur(proc_right, CONFIG['blur_kernel'], 0)
    gray_right = cv2.cvtColor(proc_right, cv2.COLOR_BGR2GRAY)
    
    # 中心线提取
    def extract_lines(gray):
        points = []
        for y in range(gray.shape[0]):
            centers = extract_centroid(gray[y])
            points.extend([[x, y] for x in centers])
        return cluster_lines(np.array(points))
    
    print("提取左图中心线...")
    left_lines = extract_lines(gray_left)
    print("提取右图中心线...")
    right_lines = extract_lines(gray_right)
    
    # 立体匹配
    print("执行匹配...")
    matched_pairs = stereo_match(left_lines, right_lines, gray_left, gray_right)
    
    # 可视化
    visualize_results(rect_left, rect_right, matched_pairs)