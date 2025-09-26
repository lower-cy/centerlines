import cv2
import numpy as np
import matplotlib.pyplot as plt

def enhance_contrast(img_gray):
    """
    使用CLAHE增强图像对比度
    参数:
        img_gray: 输入灰度图像
    返回:
        enhanced_img: 对比度增强后的图像
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(img_gray)
    return enhanced_img

def detect_and_show_edges(img, low_threshold=50, high_threshold=150):
    """
    进行边缘检测并返回提取边缘的二值化图片
    参数:
        img: 输入图像（灰度或彩色）
        low_threshold: Canny低阈值
        high_threshold: Canny高阈值
    返回:
        edges: 边缘检测结果
    """
    # 转换为灰度图
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    # 增强对比度
    img_gray = enhance_contrast(img_gray)

    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # 使用Canny算子进行边缘检测
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # 形态学操作：闭运算连接断裂的边缘
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return edges

def convert_to_grayscale(img):
    """
    将彩色图像转换为灰度图像
    参数:
        img: 输入的彩色图像
    返回:
        gray_img: 转换后的灰度图像
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def display_multiple_images(images, titles):
    """
    在一个窗口中显示多张图像
    参数:
        images: 图像列表
        titles: 对应的标题列表
    """
    num_images = len(images)
    fig, axs = plt.subplots(1, num_images, figsize=(15, 5))
    
    for i in range(num_images):
        ax = axs[i]
        if len(images[i].shape) == 3:
            ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(images[i], cmap='gray')
        ax.set_title(titles[i])
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 读取图像
    img_color = cv2.imread('R.jpg')  # 替换为彩色图像路径
    img_gray = cv2.imread('L.jpg', cv2.IMREAD_GRAYSCALE)  # 替换为灰度图像路径

    # 彩色图像转灰度图像
    color_to_gray = convert_to_grayscale(img_color)

    # 边缘检测
    edges_color = detect_and_show_edges(img_color, low_threshold=30, high_threshold=100)
    edges_gray = detect_and_show_edges(img_gray, low_threshold=30, high_threshold=100)

    # 显示所有图像
    images = [img_color, color_to_gray, img_gray, edges_color, edges_gray]
    titles = ["Original Color Image", "Converted Grayscale Image", "Original Grayscale Image", 
              "Edge Detection (Color)", "Edge Detection (Grayscale)"]
    display_multiple_images(images, titles)