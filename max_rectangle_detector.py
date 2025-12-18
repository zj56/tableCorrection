import cv2
import numpy as np
from typing import List, Optional, Tuple

class MaxRectangleDetector:
    def __init__(self, min_area_ratio: float = 0.1, aspect_ratio_range: Tuple[float, float] = (0.5, 2.0)):
        """
        初始化最大矩形检测器
        
        Args:
            min_area_ratio: 最小轮廓面积占图像面积的比例
            aspect_ratio_range: 允许的横纵比范围
        """
        self.min_area_ratio = min_area_ratio
        self.aspect_ratio_range = aspect_ratio_range
    
    def expand_polygon(self, points: np.ndarray, expansion: int = 20) -> np.ndarray:
        """
        向外扩张多边形顶点
        
        Args:
            points: 多边形顶点数组，形状为(4, 2)
            expansion: 扩张像素数
            
        Returns:
            扩张后的多边形顶点数组
        """
        # 计算多边形的中心点
        center = np.mean(points, axis=0)
        
        # 计算每个点相对于中心点的向量，并单位化
        expanded_points = []
        for point in points:
            # 计算从中心点到顶点的向量
            vector = point - center
            # 计算向量的长度
            length = np.linalg.norm(vector)
            # 如果向量长度为0，跳过（避免除以0）
            if length == 0:
                expanded_points.append(point)
                continue
            # 单位化向量
            unit_vector = vector / length
            # 沿着单位向量方向扩展指定像素
            expanded_point = point + unit_vector * expansion
            expanded_points.append(expanded_point)
        
        return np.array(expanded_points, dtype=np.int32)
    
    def detect_and_filter_contours(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """
        检测轮廓并筛选出目标轮廓

        Args:
            binary_image: 二值化图像

        Returns:
            筛选后的轮廓列表
        """
        # 检测轮廓
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 获取图像尺寸用于面积计算
        height, width = binary_image.shape
        image_area = height * width
        min_area = image_area * self.min_area_ratio

        filtered_contours = []

        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)

            # 面积过滤
            if area < min_area:
                continue

            # 获取轮廓的边界矩形
            x, y, w, h = cv2.boundingRect(contour)

            # 计算横纵比
            if h == 0:
                continue
            aspect_ratio = w / h

            # 横纵比过滤
            if (aspect_ratio < self.aspect_ratio_range[0] or 
                aspect_ratio > self.aspect_ratio_range[1]):
                continue

            filtered_contours.append(contour)

        return filtered_contours
    
    def get_largest_contour_rect(self, contours: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        获取目标轮廓的最大外接矩形
        
        Args:
            contours: 轮廓列表
            
        Returns:
            最大轮廓的近似四边形顶点（已向外扩张20像素）
        """
        if not contours:
            return None
        
        # 找到面积最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 使用多边形逼近获取四边形
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # 确保是四边形
        if len(approx) == 4:
            points = approx.reshape(4, 2)
            # 将多边形顶点向外扩张5像素（减少扩张量避免过度扩张）
            return self.expand_polygon(points, 10)
        else:
            # 如果不是四边形，使用最小外接矩形
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            points = box.astype(np.int32)
            # 将多边形顶点向外扩张0像素
            return self.expand_polygon(points, 10)
    
    def get_max_rectangle(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        从输入图像中获取最大矩形框
        
        Args:
            image: 输入图像（BGR格式）
            
        Returns:
            Tuple[处理后的图像, 最大矩形框顶点] 或 None
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊降噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 自适应阈值二值化
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 形态学操作 - 闭操作，填充小的孔洞
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 检测并筛选轮廓
        contours = self.detect_and_filter_contours(closed)
        
        # 获取最大轮廓的外接矩形
        rect_points = self.get_largest_contour_rect(contours)
        
        if rect_points is not None:
            # 在原始图像上绘制矩形
            result_image = image.copy()
            cv2.polylines(result_image, [rect_points], True, (0, 255, 0), 2)
            return result_image, rect_points
        else:
            return None

# 使用示例
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description='获取图片中最大矩形框')
    parser.add_argument('--image', type=str, default='./23.jpg', help='输入图片路径')
    parser.add_argument('--output', type=str, default='./output/max_rectangle_result.jpg', help='输出图片路径')
    args = parser.parse_args()
    
    # 读取图片
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"错误: 图片文件不存在: {image_path}")
        exit(1)
    
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"错误: 无法读取图片: {image_path}")
        exit(1)
    
    # 创建检测器实例
    detector = MaxRectangleDetector()
    
    # 获取最大矩形框
    result = detector.get_max_rectangle(image)
    
    if result:
        result_image, rect_points = result
        print(f"成功检测到最大矩形框，顶点坐标: {rect_points}")
        
        # 确保输出目录存在
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存结果图片
        cv2.imwrite(str(output_path), result_image)
        print(f"结果已保存到: {output_path}")
        
        # 显示结果
        cv2.imshow('原始图像', image)
        cv2.imshow('检测结果', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("未检测到矩形框")