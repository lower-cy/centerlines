
def extract_laser_centerline(img):
    """
    基于Hough变换的激光中心线提取算法
    参数:
        img: 输入图像（灰度或彩色）
    返回:
        center_points: 平滑后的激光中心线坐标列表[(x,y)...]
    """
    # 转换为灰度图
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    # 1. 图像预处理
    img_gray = cv2.equalizeHist(img_gray)  # 增强对比度
    binary = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 25, 7  # 调整参数以适应曲度较大的线条
    )

    # 形态学操作
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))  # 调整核大小
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))  # 调整核大小
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # 2. 过滤小区域
    filtered_binary = filter_small_regions(opened, min_area=200)

    # 3. Hough变换提取中心线
    lines = cv2.HoughLinesP(filtered_binary, 1, np.pi / 180, threshold=30, minLineLength=50, maxLineGap=20)  # 调整参数
    centers = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            centers.append(((x1 + x2) / 2, (y1 + y2) / 2))

    # 4. 后处理：插值和平滑
    if len(centers) > 10:
        centers = spline_interpolation(centers, smoothness=50)  # 调整平滑度

    return centers


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
    # 计算合适的行列数
    rows = 2
    cols = (num_images + rows - 1) // rows

    fig, axs = plt.subplots(rows, cols, figsize=(15, 10))
    
    for i in range(num_images):
        row = i // cols
        col = i % cols
        ax = axs[row, col]
        
        if len(images[i].shape) == 3:
            ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(images[i], cmap='gray')
        ax.set_title(titles[i])
        ax.axis('off')

    # 隐藏多余的子图
    for i in range(num_images, rows * cols):
        row = i // cols
        col = i % cols
        axs[row, col].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 读取图像
    img_color = cv2.imread('R.jpg')  # 替换为彩色图像路径
    img_gray = cv2.imread('L.jpg', cv2.IMREAD_GRAYSCALE)  # 替换为灰度图像路径

    # 彩色图像转灰度图像
    color_to_gray = convert_to_grayscale(img_color)

    # 边缘检测
    edges_color = detect_and_show_edges(img_color)
    edges_gray = detect_and_show_edges(img_gray)

    # 显示所有图像
    images = [img_color, color_to_gray, img_gray, edges_color, edges_gray]
    titles = ["Original Color Image", "Converted Grayscale Image", "Original Grayscale Image", "Edge Detection (Color)", "Edge Detection (Grayscale)"]
    display_multiple_images(images, titles)