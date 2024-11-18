import cv2
import numpy as np
from scipy.ndimage import center_of_mass
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

import cv2
import numpy as np
def compute_centroids_scipy(image):
    """
    使用 scipy 计算二值图像中每个物体的质心。
    """
    # 1. 转换为灰度图并二值化处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # 2. 计算连通组件
    labeled_array, num_features = ndimage.label(binary)

    # 3. 计算每个连通区域的质心
    centroids = center_of_mass(binary, labeled_array, range(1, num_features + 1))

    # 将质心转换为整数并返回为列表
    return [tuple(map(int, c)) for c in centroids]


def compute_centroids_threshold(image, min_area=100):
    """
    计算面积大于 min_area 的物体质心。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:  # 过滤掉小面积的噪声
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
    return centroids

# Use the connected components to analyze and calculate the center of mass of each object.
def compute_centroids_component(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)


    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    # exclude background
    return centroids[1:].astype(int).tolist()



# print centroids
def visualize_centroids(image, centroids):

    for (cx, cy) in centroids:

        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
