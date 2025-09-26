import cv2
import numpy as np
from skimage import filters, morphology
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

def preprocess_image(img):
    """
    预处理图像以增强对比度和减少噪声
    :param img: 输入图像
    :return: 预处理后的图像
    """
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 双边滤波
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # 高斯模糊
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    return gray

def extract_laser_center_enhanced(img):
    """
    增强版激光中心线提取函数
    :param img: 输入图像
    :return: 提取的激光中心点集
    """
    # 预处理图像
    preprocessed_img = preprocess_image(img)
    
    # 自适应阈值分割
    thresh = cv2.adaptiveThreshold(preprocessed_img, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 5)
    
    # 形态学处理
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # 开运算
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)  # 闭运算
    
    # 骨架化
    skeleton = morphology.skeletonize(cleaned / 255).astype(np.uint8) * 255
    
    # 提取初始点集
    ys, xs = np.where(skeleton > 0)
    points = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    
    return points, skeleton

def compute_normals(skeleton):
    """
    计算法线方向
    :param skeleton: 骨架化图像
    :return: 法线方向向量
    """
    # 计算梯度
    grad_x = cv2.Sobel(skeleton, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(skeleton, cv2.CV_64F, 0, 1, ksize=3)
    
    # 法线方向（垂直于梯度）
    normals = np.column_stack((-grad_y.flatten(), grad_x.flatten()))
    norms = np.linalg.norm(normals, axis=1)
    normals[norms > 0] /= norms[norms > 0][:, None]  # 归一化
    
    return normals

def subpixel_refinement(img, points, normals, step=0.5, num_steps=3):
    """
    亚像素级别精确定位
    :param img: 输入灰度图像
    :param points: 初始点集
    :param normals: 法线方向
    :param step: 插值步长
    :param num_steps: 插值范围
    :return: 亚像素级别的点集
    """
    refined_points = []
    
    for (x, y), (nx, ny) in zip(points, normals):
        # 沿法线方向采样灰度值
        samples = []
        positions = []
        
        for i in range(-num_steps, num_steps + 1):
            dx, dy = i * step * nx, i * step * ny
            px, py = x + dx, y + dy
            
            if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                val = img[int(py), int(px)]
                samples.append(val)
                positions.append((px, py))
        
        # 二次插值找到局部最大值
        if len(samples) >= 3:
            max_idx = np.argmax(samples)
            if 0 < max_idx < len(samples) - 1:
                # 二次插值公式
                a, b, c = samples[max_idx - 1], samples[max_idx], samples[max_idx + 1]
                if a != 2 * b - c:  # 避免除以零或不稳定的情况
                    try:
                        offset = 0.5 * (a - c) / (a - 2 * b + c)
                        refined_x = positions[max_idx][0] + offset * nx * step
                        refined_y = positions[max_idx][1] + offset * ny * step
                        refined_points.append((refined_x, refined_y))
                    except Exception as e:
                        print(f"Error in subpixel refinement: {e}")
    
    return np.array(refined_points)

def fit_centerline(points):
    """
    使用样条插值拟合中心线
    :param points: 提取的点集
    :return: 拟合后的中心线
    """
    if len(points) > 0:
        try:
            tck, u = splprep(points.T, s=0.1)  # 调整s参数以增加平滑度
            u_new = np.linspace(u.min(), u.max(), 1000)
            x_new, y_new = splev(u_new, tck, der=0)
            return np.column_stack((x_new, y_new))
        except Exception as e:
            print(f"Error in fit_centerline: {e}")
            return np.array([])
    else:
        return np.array([])

def process_images_enhanced(left_img_path, right_img_path):
    """
    增强版图像处理函数
    :param left_img_path: 左图像路径
    :param right_img_path: 右图像路径
    :return: 左右图像的激光中心线
    """
    # 读取图像
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)
    
    if left_img is None or right_img is None:
        print("Error: Unable to read one or both images.")
        return None, None
    
    # 提取初始点集和骨架化图像
    left_pts, left_skeleton = extract_laser_center_enhanced(left_img)
    right_pts, right_skeleton = extract_laser_center_enhanced(right_img)
    
    # 计算法线方向
    left_normals = compute_normals(left_skeleton)
    right_normals = compute_normals(right_skeleton)
    
    # 亚像素精确定位
    left_pts_refined = subpixel_refinement(cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY), left_pts, left_normals)
    right_pts_refined = subpixel_refinement(cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY), right_pts, right_normals)
    
    # 拟合中心线
    left_centerline = fit_centerline(left_pts_refined)
    right_centerline = fit_centerline(right_pts_refined)
    
    # 输出中心线坐标为二维矩阵
    if len(left_centerline) > 0:
        np.savetxt('left_centerline_subpixel.txt', left_centerline, fmt='%.4f', delimiter=',')
    if len(right_centerline) > 0:
        np.savetxt('right_centerline_subpixel.txt', right_centerline, fmt='%.4f', delimiter=',')
    
    # 可视化
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    for a, img, pts, centerline, side in zip(ax, [left_img, right_img], [left_pts_refined, right_pts_refined], [left_centerline, right_centerline], ['左', '右']):
        if len(img.shape) == 2:
            a.imshow(img, cmap='gray')
        else:
            a.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if len(pts) > 0:
            a.scatter(pts[:, 0], pts[:, 1], c='r', s=3)
        if len(centerline) > 0:
            a.plot(centerline[:, 0], centerline[:, 1], 'g-', linewidth=2)
        a.set_title(f"{side}图像 ({len(pts)}点)")
    
    plt.tight_layout()
    plt.show()
    
    return left_centerline, right_centerline

# 主程序
if __name__ == "__main__":
    left_path = 'L.jpg'
    right_path = 'R.jpg'
    
    # 处理图像
    left_centerline, right_centerline = process_images_enhanced(left_path, right_path)