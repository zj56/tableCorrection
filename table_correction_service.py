from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import DocImgOrientationClassification, TextDetection
import cv2
import numpy as np
import json
import requests
import io
import os
from pathlib import Path
import datetime
# 导入MaxRectangleDetector类
from max_rectangle_detector import MaxRectangleDetector
import uvicorn

from loguru import logger
import time
import subprocess
import sys
DEVNULL = subprocess.DEVNULL

# 配置日志
logger.remove()  # 移除默认处理器

# 从环境变量获取日志配置
log_level = os.getenv("LOG_LEVEL", "DEBUG")
log_file = os.getenv("LOG_FILE", "./logs/monitor.log")


# 控制台日志
logger.add(
    sys.stderr,
    level=log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True
)

# 文件日志（如果配置了）
log_path = Path(log_file)
log_path.parent.mkdir(parents=True, exist_ok=True)
logger.add(
    log_file,
    level=log_level,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="500 MB",
    retention="10 days",
    encoding="utf-8"
)

# 启动服务
# 初始化最大矩形检测器
logger.info("初始化最大矩形检测器")
detector = MaxRectangleDetector()
timestamp = None
# FastAPI应用
logger.info("创建FastAPI应用实例")
app = FastAPI(
    title="表格矫正服务",
    description="接收图片，返回矫正后的表格图片",
    version="1.0.0"
)

# 配置CORS
logger.info("配置CORS中间件")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化模型（在服务启动时加载一次）
logger.info("正在加载PaddleOCR模型...")
orientation_model = DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori")
text_detection_model = TextDetection(model_name="PP-OCRv5_server_det")
logger.info("模型加载完成")

# 确保output目录存在
output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    logger.info(f"创建output目录: {output_dir}")

def process_image_for_contours(image, area_threshold=50000, edge_threshold=50, timestamp=None):
    '''
    处理图像，检测轮廓并返回裁剪后的图像
    
    参数:
        image: 输入图像
        area_threshold: 面积阈值，只处理面积大于此值的轮廓
        edge_threshold: 边缘距离阈值，判断轮廓是否接近边缘
    
    返回:
        tuple: (裁剪后的图像, 带有轮廓和线条的结果图像)
    '''

    
    # 获取图像尺寸
    h, w = image.shape[:2]
    logger.info(f"处理图像轮廓，图像尺寸: {w}x{h}")


    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 进行高斯模糊去噪
    gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 进行二值化处理
    ret, binary = cv2.threshold(gaussian_blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 添加形态学膨胀操作，增强细线边框并连接不完整轮廓
    # 定义膨胀核，可以根据需要调整大小
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 膨胀操作，迭代次数设为1
    binary = cv2.dilate(binary, kernel, iterations=1)
    
    # 检测轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    logger.info(f"检测到 {len(contours)} 个轮廓")
    
    # 创建列表用于存储符合条件的边界框信息
    filtered_boxes = []
    
    # 为每个轮廓绘制边界框
    for i, contour in enumerate(contours):
        # 获取轮廓的边界框
        x, y, w_contour, h_contour = cv2.boundingRect(contour)
        
        # 计算边界框面积
        area = w_contour * h_contour
        
        # 只处理面积超过阈值的边界框
        if area > area_threshold:
            # 检查边界框内部是否为黑色区块
            # 提取边界框内的图像区域
            roi = gray[y:y+h_contour, x:x+w_contour]
            # 计算区域内的平均亮度
            mean_brightness = cv2.mean(roi)[0]
            # 如果平均亮度低于阈值，认为是黑色区块，跳过
            if mean_brightness < 50:  # 50是一个经验值，可以根据实际情况调整
                logger.debug(f"跳过边界框 {i+1}: 内部为黑色区块，平均亮度={mean_brightness}")
                continue
            
            # 检测边界框是否接近图像边缘
            is_near_edge = False
            edge_info = []
            
            # 检查是否接近左边缘
            if x < edge_threshold:
                is_near_edge = True
                edge_info.append('左边缘')
            # 检查是否接近上边缘
            if y < edge_threshold:
                is_near_edge = True
                edge_info.append('上边缘')
            # 检查是否接近右边缘
            if (x + w_contour) > (w - edge_threshold):
                is_near_edge = True
                edge_info.append('右边缘')
            # 检查是否接近下边缘
            if (y + h_contour) > (h - edge_threshold):
                is_near_edge = True
                edge_info.append('下边缘')
            
            # 记录符合条件的边界框信息
            box_info = {
                'id': i+1,
                'x': x,
                'y': y,
                'width': w_contour,
                'height': h_contour,
                'area': area,
                'top_left': (x, y),
                'bottom_right': (x + w_contour, y + h_contour),  # 修正此处的错误
                'is_near_edge': is_near_edge,
                'near_edges': edge_info
            }
            filtered_boxes.append(box_info)
    
    # 打印记录的边界框信息
    logger.info(f"符合条件的边界框数量: {len(filtered_boxes)}")
    
    # 创建一个用于绘制边界框的图像副本
    img_with_boxes = image.copy()
    
    # 在图像上绘制所有边界框
    for box in filtered_boxes:
        edge_info_text = "接近" + "、".join(box['near_edges']) if box['is_near_edge'] else "不接近边缘"
        logger.debug(f"边界框 {box['id']}: 位置=({box['x']},{box['y']}), 尺寸={box['width']}x{box['height']}, 面积={box['area']}, {edge_info_text}")
        
        # 根据是否接近边缘设置不同颜色
        if box['is_near_edge']:
            # 接近边缘的边界框用红色
            color = (0, 0, 255)  # BGR格式的红色
        else:
            # 不接近边缘的边界框用绿色
            color = (0, 255, 0)  # BGR格式的绿色
        
        # 绘制边界框
        x, y = box['x'], box['y']
        width, height = box['width'], box['height']
        cv2.rectangle(img_with_boxes, (x, y), (x + width, y + height), color, 2)
        
        # 在边界框左上角添加边界框ID
        cv2.putText(img_with_boxes, f"{box['id']}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        boxed_img_path = f"./output_contours/contours_img_{timestamp}.jpg"
        os.makedirs("./output_contours", exist_ok=True)
        cv2.imwrite(boxed_img_path, img_with_boxes)
        logger.info(f"已保存带边界框的图像到: {boxed_img_path}")
    
    # 初始化裁剪图像为原图副本
    cropped_image = image

    # 先处理右边缘（crop_left操作）
    # 找到接近右边缘的边界框
    right_edge_boxes = [box for box in filtered_boxes if '右边缘' in box['near_edges']]
    if right_edge_boxes:
        logger.info(f"接近右边缘的边界框数量: {len(right_edge_boxes)}")
        
        # 找到接近右边缘的边界框中的最左位置
        leftmost_x = min(box['x'] for box in right_edge_boxes)
        logger.info(f"接近右边缘的边界框最左位置: x={leftmost_x}")
        
        if leftmost_x < w:
            cropped_image = cropped_image[:, :leftmost_x]  # 裁剪右侧区域
        
    # 再处理左边缘（crop_right操作）在同一张图像上进行
    # 找到接近左边缘的边界框
    left_edge_boxes = [box for box in filtered_boxes if '左边缘' in box['near_edges']]
    if left_edge_boxes:
        logger.info(f"接近左边缘的边界框数量: {len(left_edge_boxes)}")
        
        # 找到接近左边缘的边界框中的最右位置
        rightmost_x = max(box['x'] + box['width'] for box in left_edge_boxes)
        logger.info(f"接近左边缘的边界框最右位置: x={rightmost_x}")

        # 直接去掉垂直线左侧的区域（裁剪）
        if rightmost_x > 0:
            cropped_image = cropped_image[:, rightmost_x:]  # 裁剪左侧区域
    
    return cropped_image

def process_image_for_text(image, timestamp=None):
    '''
    检测图片中所有的文本框，并找出所有和图片边沿接触的文本框
    同时裁剪图像，去掉接近右边缘的文本框中最靠左的边缘和图片右边缘的区域
    以及去掉接近左边缘的文本框中最靠右的边缘和图片左边缘的区域
    
    参数:
        image: 输入图像
    
    返回:
        tuple: (所有文本框列表, 与边沿接触的文本框列表, 左边缘最右位置, 右边缘最左位置, 裁剪后的图像)
    '''
    # 获取图像尺寸
    h, w = image.shape[:2]
    logger.info(f"处理文本检测，图像尺寸: 宽度={w}px, 高度={h}px")
    
    # 进行文字检测
    output_text = text_detection_model.predict(image, batch_size=1)
    
    # 创建时间戳用于生成唯一文件名

    text_json_path = f"./output_json/text_result_{timestamp}.json"
    
    # 保存检测结果到JSON文件
    for res in output_text:
        res.save_to_json(text_json_path)
    
    # 初始化文本框列表
    all_text_boxes = []
    edge_contact_boxes = []

        
    if 'dt_polys' in output_text[0] and 'dt_scores' in output_text[0]:
        dt_polys = output_text[0]['dt_polys']
        dt_scores = output_text[0]['dt_scores']
        
        # 遍历所有文本框
        for i, poly in enumerate(dt_polys):
            x_coords = [point[0] for point in poly]
            y_coords = [point[1] for point in poly]
            x1, y1 = int(min(x_coords)), int(min(y_coords))
            x2, y2 = int(max(x_coords)), int(max(y_coords))
            
            # 创建文本框信息字典
            text_box = {
                'id': i + 1,
                'bbox': (x1, y1, x2, y2),
                'width': x2 - x1,
                'height': y2 - y1,
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'score': dt_scores[i],
                'is_near_edge': False,
                'near_edges': []
            }
            
            # 检查文本框是否接近图像边沿
            # 定义接近边沿的阈值（像素）
            edge_threshold = 10
            
            # 检查是否接近左边缘
            if x1 < edge_threshold:
                text_box['is_near_edge'] = True
                text_box['near_edges'].append('左边缘')
            # 检查是否接近上边缘
            if y1 < edge_threshold:
                text_box['is_near_edge'] = True
                text_box['near_edges'].append('上边缘')
            # 检查是否接近右边缘
            if x2 > (w - edge_threshold):
                text_box['is_near_edge'] = True
                text_box['near_edges'].append('右边缘')
            # 检查是否接近下边缘
            if y2 > (h - edge_threshold):
                text_box['is_near_edge'] = True
                text_box['near_edges'].append('下边缘')
            
            # 添加到所有文本框列表
            all_text_boxes.append(text_box)
            
            # 如果接近边沿，添加到边沿接触文本框列表
            if text_box['is_near_edge']:
                edge_contact_boxes.append(text_box)
    
    # 打印检测结果
    logger.info(f"共检测到 {len(all_text_boxes)} 个文本框")
    logger.info(f"与边沿接触的文本框数量: {len(edge_contact_boxes)}")
    
    # 遍历并打印与边沿接触的文本框信息
    for box in edge_contact_boxes:
        edge_info_text = "接近" + "、".join(box['near_edges'])
        logger.debug(f"文本框 {box['id']}: 位置={box['bbox']}, 分数={box['score']:.2f}, {edge_info_text}")
    
    # 找到接近左边缘的文本框中最靠右的边缘
    left_edge_boxes = [box for box in edge_contact_boxes if '左边缘' in box['near_edges']]
    left_edge_rightmost_x = None
    if left_edge_boxes:
        left_edge_rightmost_x = max(box['bbox'][2] for box in left_edge_boxes)  # box['bbox'][2]是x2坐标
        logger.info(f"接近左边缘的文本框中最靠右的边缘位置: x={left_edge_rightmost_x}")
    else:
        logger.info("没有找到接近左边缘的文本框")
    
    # 找到接近右边缘的文本框中最靠左的边缘
    right_edge_boxes = [box for box in edge_contact_boxes if '右边缘' in box['near_edges']]
    right_edge_leftmost_x = None
    if right_edge_boxes:
        right_edge_leftmost_x = min(box['bbox'][0] for box in right_edge_boxes)  # box['bbox'][0]是x1坐标
        logger.info(f"接近右边缘的文本框中最靠左的边缘位置: x={right_edge_leftmost_x}")
    else:
        logger.info("没有找到接近右边缘的文本框")
    
    # 创建图像副本用于裁剪
    cropped_image = image
    
    # 在原始图像上标注与边缘接触的文本框
    annotated_image = image.copy()
    
    # 为每个与边缘接触的文本框绘制边界框和标签
    for box in edge_contact_boxes:
        x1, y1, x2, y2 = box['bbox']
        edge_info_text = "接近" + "、".join(box['near_edges'])
        label = f"{box['id']}: {edge_info_text}"
        
        # 绘制文本框边界框（蓝色）
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 在文本框上方绘制标签
        # 计算标签背景的高度和宽度
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # 确保标签不会超出图像边界
        label_y = max(y1 - 10, text_height + 5)
        
        # 绘制标签背景（白色）
        cv2.rectangle(annotated_image, (x1, label_y - text_height - baseline), 
                     (x1 + text_width, label_y), (255, 255, 255), -1)
        
        # 绘制标签文本（黑色）
        cv2.putText(annotated_image, label, (x1, label_y - baseline), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 保存标注后的图像
    output_dir = "./output_text"
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用时间戳生成唯一的文件名
    annotated_img_path = f"{output_dir}/annotated_text_edges_{timestamp}.jpg"
    cv2.imwrite(annotated_img_path, annotated_image)
    logger.info(f"已保存标注与边缘接触文本框的图像到: {annotated_img_path}")
    
    # 先处理右边缘
    # 去掉接近右边缘的文本框中最靠左的边缘和图片右边缘的区域
    if right_edge_leftmost_x is not None:
        # print(f"裁剪右边缘: 从{x=right_edge_leftmost_x}到图像右边缘")
        # 直接去掉右边缘区域（裁剪）
        # 确保使用当前图像的实际宽度进行比较
        h_current, w_current = cropped_image.shape[:2]
        if right_edge_leftmost_x < w_current:
            logger.info(f"裁剪右边缘区域，从x={right_edge_leftmost_x}到图像右边缘")
            cropped_image = cropped_image[:, :right_edge_leftmost_x]  # 裁剪右侧区域
    
    # 再处理左边缘
    # 去掉接近左边缘的文本框中最靠右的边缘和图片左边缘的区域
    # 注意：这里需要重新计算图像尺寸，因为可能已经进行了右边缘裁剪
    h_cropped, w_cropped = cropped_image.shape[:2]
    if left_edge_rightmost_x is not None:
        # print(f"裁剪左边缘: 从图像左边缘到{x=left_edge_rightmost_x}")
        # 直接去掉左边缘区域（裁剪）
        if left_edge_rightmost_x > 0 and left_edge_rightmost_x < w_cropped:
            logger.info(f"裁剪左边缘区域，从图像左边缘到x={left_edge_rightmost_x}")
            cropped_image = cropped_image[:, left_edge_rightmost_x:]  # 裁剪左侧区域
    
    # 返回结果，包含所有文本框、边缘接触文本框、左边缘最右位置、右边缘最左位置和裁剪后的图像
    return  cropped_image



def remove_edge_adjacent_table(image, area_ratio_threshold=0.05, edge_threshold=30, timestamp=None):
    """
    找到与图像边缘贴近的半幅表格轮廓，从其内侧竖线/横线处裁剪，
    保留剩余的主体内容部分（去掉贴边的不完整表格）。

    策略：
      1. 灰度 → 高斯模糊 → Otsu 二值化 + 膨胀，连通细线
      2. 查找外轮廓，只保留面积 > area_ratio_threshold * 图像面积 的轮廓
      3. 保留「至少一条边与图像边缘距离 < edge_threshold」的轮廓
      4. 按贴边方向找内侧边线，依次裁剪（去掉贴边区域，留下主体）：
           - 右侧贴边表格：找所有右侧贴边轮廓的最小 x（内边线），裁剪为 [:, :min_x]
           - 左侧贴边表格：找所有左侧贴边轮廓的最大 x2（内边线），裁剪为 [:, max_x2:]
           - 下侧贴边表格：找所有下侧贴边轮廓的最小 y（内边线），裁剪为 [:min_y, :]
           - 上侧贴边表格：找所有上侧贴边轮廓的最大 y2（内边线），裁剪为 [max_y2:, :]

    参数:
        image                : 输入 BGR 图像
        area_ratio_threshold : 面积阈值（相对图像面积的比例）
        edge_threshold       : 判断是否贴近边缘的像素距离阈值
        timestamp            : 日志/调试文件时间戳

    返回:
        cropped_image : 去掉贴边表格后的图像；若未找到贴边矩形则返回 None
    """
    h, w = image.shape[:2]
    min_area = area_ratio_threshold * h * w
    logger.info(f"[RemoveEdge] 图像尺寸={w}x{h}, 最小面积阈值={min_area:.0f}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.info(f"[RemoveEdge] 检测到轮廓数量: {len(contours)}")

    # 按贴边方向分桶存储轮廓 bounding box
    right_boxes  = []  # 贴右边缘
    left_boxes   = []  # 贴左边缘
    bottom_boxes = []  # 贴下边缘
    top_boxes    = []  # 贴上边缘

    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < min_area:
            continue
        # 排除内部平均亮度极低的纯黑区块（装订孔等）
        roi = gray[by:by+bh, bx:bx+bw]
        if cv2.mean(roi)[0] < 40:
            continue
        near_left   = bx < edge_threshold
        near_top    = by < edge_threshold
        near_right  = (bx + bw) > (w - edge_threshold)
        near_bottom = (by + bh) > (h - edge_threshold)
        if not (near_left or near_top or near_right or near_bottom):
            continue
        edges = []
        if near_right:  right_boxes.append((bx, by, bx+bw, by+bh));  edges.append('右')
        if near_left:   left_boxes.append((bx, by, bx+bw, by+bh));    edges.append('左')
        if near_bottom: bottom_boxes.append((bx, by, bx+bw, by+bh));  edges.append('下')
        if near_top:    top_boxes.append((bx, by, bx+bw, by+bh));     edges.append('上')
        logger.info(f"[RemoveEdge] 贴边轮廓: ({bx},{by})-({bx+bw},{by+bh}), 贴近: {edges}")

    if not (right_boxes or left_boxes or bottom_boxes or top_boxes):
        logger.info("[RemoveEdge] 未找到任何贴边大矩形")
        return None

    # 确定四个方向的裁剪边界（默认保留整幅图）
    cut_x2 = w   # 右侧裁剪：保留 [:, :cut_x2]
    cut_x1 = 0   # 左侧裁剪：保留 [:, cut_x1:]
    cut_y2 = h   # 下侧裁剪：保留 [:cut_y2, :]
    cut_y1 = 0   # 上侧裁剪：保留 [cut_y1:, :]

    if right_boxes:
        # 右侧贴边表格的内边线 = 所有右侧贴边轮廓的最小 x
        inner_x = min(b[0] for b in right_boxes)
        cut_x2 = inner_x
        logger.info(f"[RemoveEdge] 右侧贴边表格，内边线 x={inner_x}，裁剪保留 [:, :{cut_x2}]")

    if left_boxes:
        # 左侧贴边表格的内边线 = 所有左侧贴边轮廓的最大 x2
        inner_x2 = max(b[2] for b in left_boxes)
        cut_x1 = inner_x2
        logger.info(f"[RemoveEdge] 左侧贴边表格，内边线 x2={inner_x2}，裁剪保留 [:, {cut_x1}:]")

    if bottom_boxes:
        # 下侧贴边表格的内边线 = 所有下侧贴边轮廓的最小 y
        inner_y = min(b[1] for b in bottom_boxes)
        cut_y2 = inner_y
        logger.info(f"[RemoveEdge] 下侧贴边表格，内边线 y={inner_y}，裁剪保留 [:{cut_y2}, :]")

    if top_boxes:
        # 上侧贴边表格的内边线 = 所有上侧贴边轮廓的最大 y2
        inner_y2 = max(b[3] for b in top_boxes)
        cut_y1 = inner_y2
        logger.info(f"[RemoveEdge] 上侧贴边表格，内边线 y2={inner_y2}，裁剪保留 [{cut_y1}:, :]")

    # 合理性检查
    if cut_x1 >= cut_x2 or cut_y1 >= cut_y2:
        logger.warning(f"[RemoveEdge] 裁剪范围无效: x=[{cut_x1},{cut_x2}], y=[{cut_y1},{cut_y2}]，返回 None")
        return None

    # 调试：保存标注图
    if timestamp:
        dbg = image.copy()
        for b in right_boxes + left_boxes + bottom_boxes + top_boxes:
            cv2.rectangle(dbg, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cv2.rectangle(dbg, (cut_x1, cut_y1), (cut_x2, cut_y2), (0, 255, 0), 3)
        os.makedirs("./output_edge_rect", exist_ok=True)
        cv2.imwrite(f"./output_edge_rect/{timestamp}.jpg", dbg)
        logger.info(f"[RemoveEdge] 调试图已保存至 ./output_edge_rect/{timestamp}.jpg")

    cropped = image[cut_y1:cut_y2, cut_x1:cut_x2]
    logger.info(f"[RemoveEdge] 裁剪结果尺寸: {cropped.shape[1]}x{cropped.shape[0]}")
    return cropped


def remove_binding_holes(img_bgr, table_x1, table_y1, table_x2, table_y2):
    """
    使用霍夫圆检测，去除主表格矩形框外侧的装订孔（黑色圆点）。
    检测到的圆点区域将用白色填充。

    参数:
        img_bgr  : 输入图像（BGR）
        table_x1 : 主表格左边界（在 img_bgr 坐标系中）
        table_y1 : 主表格上边界
        table_x2 : 主表格右边界
        table_y2 : 主表格下边界
    返回:
        去除装订孔后的图像副本
    """
    h, w = img_bgr.shape[:2]
    result = img_bgr.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 只扫描左右两侧区域：装订孔仅出现在左/右侧边缘
    # 不扫描上下区域，避免将页码等内容误判为装订孔
    outer_regions = [
        (0,        0, table_x1, h),   # 左侧
        (table_x2, 0, w,        h),   # 右侧
    ]

    total_removed = 0
    for (rx1, ry1, rx2, ry2) in outer_regions:
        if rx2 <= rx1 or ry2 <= ry1:
            continue
        roi_gray = gray[ry1:ry2, rx1:rx2]
        # 高斯模糊降噪，有助于霍夫圆检测
        blurred = cv2.GaussianBlur(roi_gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=25,
            minRadius=10,
            maxRadius=60,
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            for (cx, cy, r) in circles:
                abs_cx = rx1 + int(cx)
                abs_cy = ry1 + int(cy)
                # 扩大半径，确保完整覆盖圆点边缘及残留黑色像素
                cv2.circle(result, (abs_cx, abs_cy), int(r)+18, (255, 255, 255), -1)
                total_removed += 1
                logger.info(f"装订孔已去除: 圆心=({abs_cx},{abs_cy}), 半径={r}")

    logger.info(f"霍夫圆检测共去除装订孔: {total_removed} 个")
    return result


def correct_image_orientation(img, orientation_model, timestamp=None):
        '''
        检测图像方向并进行旋转纠正
        
        参数:
            img: 输入图像
            orientation_model: PaddleOCR方向检测模型实例
        
        返回:
            rotated_img: 旋转纠正后的图像
        '''
        output = orientation_model.predict(img, batch_size=1)
        rotated_img = None
        
        json_file_path = f"./output_json/result_{timestamp}.json"
        for res in output:
            res.save_to_json(json_file_path)
        # 直接从output中获取旋转角度信息
        if output and len(output) > 0:
            # 根据提供的output示例，它是一个包含字典的列表
            if isinstance(output[0], dict):
                res = output[0]
                # 直接从字典中获取所需信息
                if 'label_names' in res and res['label_names'] and 'scores' in res and res['scores'] and float(res['scores'][0]) > 0.7:
                    rotation_angle = int(res['label_names'][0])
                    logger.info(f"检测到旋转角度: {rotation_angle}")
                    logger.info(f"置信度: {float(res['scores'][0])}")
                    # 根据旋转角度执行相应的旋转操作
                    if rotation_angle == 90:
                        logger.info("执行逆时针90度旋转")
                        rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif rotation_angle == 180:
                        logger.info("执行180度旋转")
                        rotated_img = cv2.rotate(img, cv2.ROTATE_180)
                    elif rotation_angle == 270:
                        logger.info("执行顺时针90度旋转")
                        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    elif rotation_angle != 0:
                        logger.info(f"执行{rotation_angle}度旋转")
                        (h, w) = img.shape[:2]
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                        rotated_img = cv2.warpAffine(img, M, (w, h))
                    else:
                        logger.info("图像方向正确，无需旋转")
                        rotated_img = img.copy()
        
        # 如果没有成功获取旋转角度或执行旋转，返回原图副本
        if rotated_img is None:
            logger.warning("未能获取有效的旋转角度，使用原始图像")
            rotated_img = img.copy()
        
        logger.info("图像方向校正完成")
        return rotated_img

@app.post("/correct_detection_table")
async def correct_detection_table(
    file: UploadFile = File(...),
):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    logger.info(f"接收到新的图像矫正请求，时间戳: {timestamp}")
    """接收图片，返回矫正后的表格图片"""
    try:
        
        # 初始化变量，确保即使发生异常也不会导致后续代码出错
        table_regions = []
        rotated_img = None
        
        # 1. 读取上传的图片
        img_bytes = await file.read()
        # 检查图像数据是否为空
        if not img_bytes:
            logger.error("上传的图片数据为空")
            raise HTTPException(status_code=400, detail="上传的图片数据为空")
        logger.info(f"成功读取图片数据，大小: {len(img_bytes)} 字节")
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error("无法读取上传的图片")
            raise HTTPException(status_code=400, detail="无法读取上传的图片")
        logger.info(f"成功解码图片，尺寸: {img.shape[1]}x{img.shape[0]}")
     
        
        # 调用函数进行图像方向纠正
        rotated_img = correct_image_orientation(img, orientation_model, timestamp=timestamp)
        os.makedirs(f"./output_orientation/", exist_ok=True)
        cv2.imwrite(f"./output_orientation/{timestamp}.jpg", rotated_img)
        logger.info(f"已保存方向校正后的图像到: ./output_orientation/{timestamp}.jpg")
        # 3. 使用MaxRectangleDetector获取最大表格区域
        try:
            # 获取最大矩形框（模拟表格区域）
            # get_max_rectangle返回的是(处理后的图像, 最大矩形框顶点)的元组
            logger.info("尝试使用MaxRectangleDetector获取最大表格区域")
            result = detector.get_max_rectangle(rotated_img)

            if result is not None:
                logger.info("成功获取最大表格区域")
                result_image, rect_points = result
                table_regions = []         
                # 计算矩形的边界框坐标
                # 从rect_points中提取x和y坐标
                x_coords = rect_points[:, 0]
                y_coords = rect_points[:, 1]
                x1 = int(min(x_coords))
                y1 = int(min(y_coords))
                x2 = int(max(x_coords))
                y2 = int(max(y_coords))
                
                # 确保坐标在图像范围内
                h, w = rotated_img.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                rect_w = x2 - x1
                rect_h = y2 - y1
                logger.info(f"[诊断] 主矩形框坐标: x1={x1}, y1={y1}, x2={x2}, y2={y2}, 图像尺寸={w}x{h}")
                logger.info(f"[诊断] 左侧扫描宽度: {x1}px, 右侧扫描宽度: {w - x2}px")

                # 在裁剪前去除主矩形框外侧的装订孔
                # 此时图像完整，装订孔是完整圆形，HoughCircles 效果最佳
                rotated_img = remove_binding_holes(rotated_img, x1, y1, x2, y2)
                logger.info("已在裁剪前完成装订孔去除")

                # -------------------------------------------------------
                # 检测主矩形框外侧是否存在"半幅表格"
                # 逻辑：扫描 x1 左侧 和 x2 右侧区域，寻找与主矩形高度相近
                # 的长竖线（即半幅表格的外边线）。
                # 检测方法：对该区域做灰度→二值化→形态学竖线提取→按列求和。
                # -------------------------------------------------------
                # -------------------------------------------------------
                # 检测方法改为「三段像素密度」投影：
                # 将扫描区域的 y 范围均分为上、中、下三段，分别统计每列的
                # 暗像素占比。如果某列在全部三段的暗像素比例均 ≥ 阈值，
                # 则该列为贯通上下的竖直边缘（容忍中间有缺口/切口的情况）。
                # -------------------------------------------------------
                half_table_found = False
                half_table_side = None  # 'left' or 'right'
                # 每段内某列暗像素占该段高度的最低比例（可调整）
                ZONE_DARK_RATIO = 0.05  # 降低阈值，提高对浅色线条的灵敏度
                ZONE_MIN_SATISFY = 2    # 至少满足的段数（3段中至少2段），容忍半幅表格不覆盖全高的情况

                def detect_vertical_lines_in_region(img_bgr, rx1, rx2, ry1, ry2):
                    """
                    三段密度检测：将 [ry1, ry2] 分成上、中、下三段，
                    对每段做灰度→二值化→按列求暗像素比例。
                    若某列在至少 ZONE_MIN_SATISFY 段中 ≥ ZONE_DARK_RATIO，
                    认为是竖直边线（容忍半幅表格未覆盖全高的情况）。
                    返回 (是否找到, 符合列的原图 x 坐标列表)。
                    """
                    if rx1 >= rx2 or ry1 >= ry2:
                        return False, []
                    zone_h = ry2 - ry1
                    third = zone_h // 3
                    zones = [
                        (ry1,                ry1 + third),          # 上段
                        (ry1 + third,        ry1 + 2 * third),      # 中段
                        (ry1 + 2 * third,    ry2),                  # 下段
                    ]
                    # 各段的暗像素比例矩阵，shape = (n_zones, width)
                    zone_ratios = []
                    zone_names = ["上段", "中段", "下段"]
                    for idx, (zs, ze) in enumerate(zones):
                        roi = img_bgr[zs:ze, rx1:rx2]
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        # 简单阈值：<180 视为暗（线条像素）
                        dark_mask = (gray < 180).astype(np.float32)
                        ratio = dark_mask.mean(axis=0)  # shape: (width,)
                        zone_ratios.append(ratio)
                        logger.debug(f"[诊断] 区域x=[{rx1},{rx2}] {zone_names[idx]} 最大暗像素密度: {ratio.max():.4f} (阈值={ZONE_DARK_RATIO})")
                    # 统计每列满足阈值的段数，至少 ZONE_MIN_SATISFY 段即认为是竖线
                    satisfy_count = np.zeros(rx2 - rx1, dtype=int)
                    for r in zone_ratios:
                        satisfy_count += (r >= ZONE_DARK_RATIO).astype(int)
                    combined = satisfy_count >= ZONE_MIN_SATISFY
                    cols = np.where(combined)[0]
                    abs_cols = [rx1 + int(c) for c in cols]
                    logger.debug(f"[诊断] 区域x=[{rx1},{rx2}] 满足条件的列数: {len(abs_cols)}")
                    return len(abs_cols) > 0, abs_cols

                # 扫描左侧区域（从 0 到 x1）
                found_left, left_line_cols = detect_vertical_lines_in_region(
                    rotated_img, 0, x1, y1, y2
                )
                # 扫描右侧区域（从 x2 到 图像右边缘）
                img_h, img_w = rotated_img.shape[:2]
                found_right, right_line_cols = detect_vertical_lines_in_region(
                    rotated_img, x2, img_w, y1, y2
                )

                if found_left:
                    half_table_found = True
                    half_table_side = 'left'
                    logger.info(f"检测到左侧半幅表格，外边线 x 坐标: {left_line_cols}")
                if found_right:
                    half_table_found = True
                    half_table_side = ('both' if half_table_side == 'left' else 'right')
                    logger.info(f"检测到右侧半幅表格，外边线 x 坐标: {right_line_cols}")

                if half_table_found:
                    # 存在半幅表格，按最靠近主矩形的外边线裁剪（含去除外边线本身）：
                    # 左侧：取 left_line_cols 最大值（最靠近 x1）+ 1，从该点向右裁剪
                    # 右侧：取 right_line_cols 最小值（最靠近 x2），裁剪到该点（不含）
                    if found_left and found_right:
                        crop_x1 = max(left_line_cols) + 1
                        crop_x2 = min(right_line_cols)
                        logger.info(f"两侧均有半幅表格，从左最近外边线x={crop_x1}截到右最近外边线x={crop_x2}")
                    elif found_left:
                        crop_x1 = max(left_line_cols) + 1
                        crop_x2 = img_w
                        logger.info(f"左侧有半幅表格，从最近外边线x={crop_x1}截到右页边缘x={crop_x2}")
                    else:  # found_right only
                        crop_x1 = 0
                        crop_x2 = min(right_line_cols)
                        logger.info(f"右侧有半幅表格，从左页边缘x=0截到最近外边线x={crop_x2}")

                    crop_y1 = 0
                    crop_y2 = img_h
                    cropped_img = rotated_img[crop_y1:crop_y2, crop_x1:crop_x2]
                    logger.info(f"裁剪后区域: 位置=({crop_x1},{crop_y1}), 尺寸={crop_x2-crop_x1}x{crop_y2-crop_y1}")

                    # 在裁剪后的图像中，重新计算主表格矩形框的坐标（相对于裁剪后坐标系）
                    adj_x1 = max(0, x1 - crop_x1)
                    adj_x2 = max(0, x2 - crop_x1)
                    adj_y1 = max(0, y1 - crop_y1)
                    adj_y2 = max(0, y2 - crop_y1)
                    logger.info(f"裁剪后主表格区域（新坐标系）: ({adj_x1},{adj_y1})-({adj_x2},{adj_y2})")

                    is_success, buffer = cv2.imencode(".jpg", cropped_img)
                    if not is_success:
                        logger.error("无法将截取的图像编码为JPEG格式")
                        raise Exception("无法将截取的图像编码为JPEG格式")

                    os.makedirs("./output", exist_ok=True)
                    cropped_img_path = f"./output/{timestamp}.jpg"
                    cv2.imwrite(cropped_img_path, cropped_img)
                    logger.info(f"已截取区域并保存到: {cropped_img_path}")

                    return StreamingResponse(
                        io.BytesIO(buffer.tobytes()),
                        media_type="image/jpeg"
                    )
                else:
                    # 未发现相邻半幅表格；用 remove_edge_adjacent_table 再扫描一次
                    # 目的：去掉紧贴图像边缘的不完整表格，保留剩余主体内容
                    logger.info("未检测到相邻半幅表格，尝试用轮廓法去除贴边不完整表格")
                    removed = remove_edge_adjacent_table(rotated_img, timestamp=timestamp)
                    if removed is not None:
                        logger.info("成功去除贴边表格，返回剩余主体")
                        os.makedirs("./output", exist_ok=True)
                        cv2.imwrite(f"./output/{timestamp}.jpg", removed)
                        is_success, buffer = cv2.imencode(".jpg", removed)
                        if not is_success:
                            logger.error("无法将裁剪图像编码为JPEG格式")
                            raise HTTPException(status_code=500, detail="无法将图像编码为JPEG格式")
                        return StreamingResponse(
                            io.BytesIO(buffer.tobytes()),
                            media_type="image/jpeg"
                        )
                    else:
                        # 轮廓法也未找到贴边表格，返回完整校正图
                        logger.info("未找到贴边表格，返回矫正后的原图")
                        img_to_encode = rotated_img if rotated_img is not None and len(rotated_img.shape) > 1 else img
                        is_success, buffer = cv2.imencode(".jpg", img_to_encode)
                        if not is_success:
                            logger.error("无法将图像编码为JPEG格式")
                            raise HTTPException(status_code=500, detail="无法将图像编码为JPEG格式")
                        return StreamingResponse(
                            io.BytesIO(buffer.tobytes()),
                            media_type="image/jpeg"
                        )
            else:
                # get_max_rectangle 未能找到最大矩形框
                # 退而求其次：用轮廓法检测贴边的不完整表格，去掉它，保留剩余主体
                logger.warning("未能获取最大矩形框，尝试用轮廓法去除贴边不完整表格")
                edge_img = rotated_img if rotated_img is not None and len(rotated_img.shape) > 1 else img
                removed = remove_edge_adjacent_table(edge_img, timestamp=timestamp)
                if removed is not None:
                    logger.info("成功去除贴边表格，返回剩余主体")
                    os.makedirs("./output", exist_ok=True)
                    cv2.imwrite(f"./output/{timestamp}.jpg", removed)
                    is_success, buffer = cv2.imencode(".jpg", removed)
                    if not is_success:
                        logger.error("无法将裁剪图像编码为JPEG格式")
                        raise HTTPException(status_code=500, detail="无法将图像编码为JPEG格式")
                    return StreamingResponse(
                        io.BytesIO(buffer.tobytes()),
                        media_type="image/jpeg"
                    )
                else:
                    logger.warning("未找到贴边表格，直接返回原图")
                    is_success, buffer = cv2.imencode(".jpg", edge_img)
                    if not is_success:
                        logger.error("无法将图像编码为JPEG格式")
                        raise HTTPException(status_code=500, detail="无法将图像编码为JPEG格式")
                    return StreamingResponse(
                        io.BytesIO(buffer.tobytes()),
                        media_type="image/jpeg"
                    )
                
        except ImportError:
            logger.error("无法导入MaxRectangleDetector，请确保max_rectangle_detector.py文件存在")
            raise HTTPException(status_code=500, detail="无法导入MaxRectangleDetector，请确保max_rectangle_detector.py文件存在")
        except Exception as e:
            # 发生错误，直接返回原图
            logger.error(f"获取最大表格区域时出错: {str(e)}，将返回原图")
            # 检查rotated_img是否为空，为空时使用原始图像
            img_to_encode = rotated_img if rotated_img is not None and len(rotated_img.shape) > 1 else img
            # 将原图转换为可流式传输的格式
            is_success, buffer = cv2.imencode(".jpg", img_to_encode)
            if not is_success:
                logger.error("无法将图像编码为JPEG格式")
                raise HTTPException(status_code=500, detail="无法将图像编码为JPEG格式")
            
            # 返回原始图像
            return StreamingResponse(
                io.BytesIO(buffer.tobytes()),
                media_type="image/jpeg"
            )
    except Exception as e:
        # 发生错误，直接返回原图
        logger.error(f"处理图像时出错: {str(e)}，将返回原图")
        # 检查rotated_img是否为空，为空时使用原始图像
        img_to_encode = rotated_img if rotated_img is not None and len(rotated_img.shape) > 1 else img
        # 将原图转换为可流式传输的格式
        is_success, buffer = cv2.imencode(".jpg", img_to_encode)
        if not is_success:
            logger.error("无法将图像编码为JPEG格式")
            raise HTTPException(status_code=500, detail="无法将图像编码为JPEG格式")
        
        # 返回图像
        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/jpeg"
        )
 
    



@app.get("/")
async def root():
    """服务根路径，返回API信息"""
    logger.info("收到服务根路径请求")
    return {
        "service": "表格矫正服务",
        "version": "1.0.0",
        "endpoint": "/correct_detection_table",
        "method": "POST",
        "description": "接收图片，返回矫正后的表格图片"
    }

if __name__ == "__main__":
    logger.info("启动表格矫正服务")
    uvicorn.run(
        "table_correction_service:app",
        host="0.0.0.0",
        port=8001,  # 使用不同的端口避免冲突
        reload=True,
        log_level="info"
    )