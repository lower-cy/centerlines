import cv2
import numpy as np
from skimage import filters, morphology
import matplotlib.pyplot as plt


def preprocess_image(img):
    """
    预处理图像以增强对比度和减少噪声
    :param img: 输入图像
    :return: 预处理后的图像
    """
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # 高斯模糊
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    
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
    
    return points


def process_images_enhanced(left_img_path, right_img_path):
    """
    增强版图像处理函数
    :param left_img_path: 左图像路径
    :param right_img_path: 右图像路径
    :return: 左右图像的激光中心点集
    """
    # 读取图像
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)
    
    # 提取中心点
    left_pts = extract_laser_center_enhanced(left_img)
    right_pts = extract_laser_center_enhanced(right_img)
    
    # 输出中心线坐标为二维矩阵
    np.savetxt('left_centerline.txt', left_pts, fmt='%.2f', delimiter=',')
    np.savetxt('right_centerline.txt', right_pts, fmt='%.2f', delimiter=',')
    
    # 可视化
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    for a, img, pts, side in zip(ax, [left_img, right_img], [left_pts, right_pts], ['左', '右']):
        if len(img.shape) == 2:
            a.imshow(img, cmap='gray')
        else:
            a.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if len(pts) > 0:
            a.scatter(pts[:, 0], pts[:, 1], c='r', s=3)
        a.set_title(f"{side}图像 ({len(pts)}点)")
    
    plt.tight_layout()
    plt.show()
    
    return left_pts, right_pts


# 主程序
if __name__ == "__main__":
    # 替换为实际图像路径
    left_path = 'L.bmp'
    right_path = 'R.jpg'
    
    # 处理图像
    left_points, right_points = process_images_enhanced(left_path, right_path)