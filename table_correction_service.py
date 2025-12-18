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
                
                # 将最大表格区域加入到table_regions
                max_table_region = {
                    'bbox': (x1, y1, x2, y2),
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'width': x2 - x1,
                    'height': y2 - y1
                }
                table_regions.append(max_table_region)
                logger.info(f"检测到表格区域: 位置=({x1},{y1}), 尺寸={x2-x1}x{y2-y1}")
            else:
                logger.warning("未能使用MaxRectangleDetector找到表格区域，尝试使用轮廓和文本检测方法")
                # 处理图像时确保图像不为空
                processed_img = process_image_for_contours(rotated_img, timestamp=timestamp)
                os.makedirs(f"./output_contours/", exist_ok=True)
                # 检查处理后的图像是否为空
                if processed_img is not None and len(processed_img.shape) > 1:
                    cv2.imwrite(f"./output_contours/{timestamp}.jpg", processed_img)
                else:
                    # 如果处理后的图像为空，使用原始图像
                    cv2.imwrite(f"./output_contours/{timestamp}.jpg", rotated_img)
                # 更新rotated_img，确保不为空
                rotated_img = processed_img if processed_img is not None and len(processed_img.shape) > 1 else rotated_img
                logger.info(f"已保存轮廓处理后的图像到: ./output_contours/{timestamp}.jpg")
                cropped_image =  process_image_for_text(rotated_img, timestamp=timestamp)
                os.makedirs(f"./output_text", exist_ok=True)
                cv2.imwrite(f"./output_text/{timestamp}.jpg", cropped_image)
                logger.info(f"已保存文本处理后的图像到: ./output_text/{timestamp}.jpg")
                # 对result_image进行文字检测
                output_text = text_detection_model.predict(cropped_image, batch_size=1)
                text_regions = []
                text_json_path = f"./output_json/{timestamp}.json"
                
                for res in output_text:
                    res.save_to_json(text_json_path)
               
                         
                if 'dt_polys' in output_text[0] and 'dt_scores' in output_text[0]:
                    dt_polys = output_text[0]['dt_polys']
                    dt_scores = output_text[0]['dt_scores']
                    
                    for i, poly in enumerate(dt_polys):
                        x_coords = [point[0] for point in poly]
                        y_coords = [point[1] for point in poly]
                        x1, y1 = int(min(x_coords)), int(min(y_coords))
                        x2, y2 = int(max(x_coords)), int(max(y_coords))
                        
                        text_regions.append({
                            'bbox': (x1, y1, x2, y2),
                            'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                            'score': dt_scores[i]
                        })
                
                # 在result_image上绘制文本区域并保存
                if text_regions:
                    # img_with_text_boxes = cropped_image
                    
                    # 初始化边界区域
                    min_x = float('inf')
                    min_y = float('inf')
                    max_x = 0
                    max_y = 0
                    
                    # 遍历所有文本框，找到整体边界
                    for region in text_regions:
                        x1, y1, x2, y2 = region['bbox']
                        # cv2.rectangle(img_with_text_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # 更新边界值
                        min_x = min(min_x, x1)
                        min_y = min(min_y, y1)
                        max_x = max(max_x, x2)
                        max_y = max(max_y, y2)
                    
                    
                    # 截取整体文本区域
                    if min_x < max_x and min_y < max_y:
                        # 确保坐标在图像范围内
                        h, w = cropped_image.shape[:2]
                        min_x = max(0, min_x)
                        min_y = max(0, min_y)
                        max_x = min(w, max_x)
                        max_y = min(h, max_y)
                        
                        # 截取区域
                        cropped_text_region = cropped_image[min_y:max_y, min_x:max_x]
                        
                        # 保存截取的文本区域
                        text_region_path = f"./output/text_region_{timestamp}.jpg"
                        cv2.imwrite(text_region_path, cropped_text_region)
                        logger.info(f"已截取文本区域并保存到: {text_region_path}")
                
                is_success, buffer = cv2.imencode(".jpg", cropped_text_region)
                if not is_success:
                    logger.error("无法将图像编码为JPEG格式")
                    raise HTTPException(status_code=500, detail="无法将图像编码为JPEG格式")
                
                logger.info("成功处理图像请求，返回处理后的图像")
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
 
    
    # 在原始旋转图像上绘制mask区域的矩形框并保存
    # if table_regions:
    #     # 获取第一个表格区域的边界框坐标
    #     table_region = table_regions[0]
    #     x1, y1, x2, y2 = table_region['bbox']
        
        # # 创建原始旋转图像的副本
        # img_with_box = rotated_img.copy()
        
        # # 在图像上绘制红色矩形框，线宽为2
        # cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # # 使用时间戳生成唯一文件名

        # boxed_img_path = f"./output_mask/boxed_img_{timestamp}.jpg"
        # os.makedirs("./output_mask", exist_ok=True)
        # cv2.imwrite(boxed_img_path, img_with_box)
        # print(f"已保存带矩形框的图像到: {boxed_img_path}")
    
    output_text = text_detection_model.predict(rotated_img, batch_size=1)
    text_regions = []
    text_json_path = f"./output_json/text_result_{timestamp}.json"
    
    for res in output_text:
        res.save_to_json(text_json_path)
    

            
    if 'dt_polys' in output_text[0] and 'dt_scores' in output_text[0]:
        dt_polys = output_text[0]['dt_polys']
        dt_scores = output_text[0]['dt_scores']
        
        for i, poly in enumerate(dt_polys):
            x_coords = [point[0] for point in poly]
            y_coords = [point[1] for point in poly]
            x1, y1 = int(min(x_coords)), int(min(y_coords))
            x2, y2 = int(max(x_coords)), int(max(y_coords))
            
            text_regions.append({
                'bbox': (x1, y1, x2, y2),
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'score': dt_scores[i]
            })
    
    # 6. 查找表格标题并截取区域
    try:
        if table_regions and text_regions:
            logger.info("开始处理表格区域和文本区域")
            # 处理第一个表格（可以扩展为处理多个表格）
            table_region = table_regions[0]
            table_x1, table_y1, table_x2, table_y2 = table_region['bbox']
            
            # 收集符合条件的文本区域
            matched_text_regions = []
            for text_region in text_regions:
                text_center_x, text_center_y = text_region['center']
                if table_x1 <= text_center_x <= table_x2:
                    matched_text_regions.append(text_region)
            
            # 确定纵坐标范围
            if matched_text_regions:
                top_y1 = min(min([text['bbox'][1] for text in matched_text_regions]), table_y1)
                bottom_y2 = max(max([text['bbox'][3] for text in matched_text_regions]), table_y2)
                logger.info(f"找到{len(matched_text_regions)}个匹配的文本区域，确定纵坐标范围: {top_y1}-{bottom_y2}")
            else:
                top_y1 = table_y1
                bottom_y2 = table_y2
                logger.info("未找到匹配的文本区域，使用表格区域的纵坐标范围")
            
            # 截取区域
            h, w = rotated_img.shape[:2]
            crop_x1 = max(0, table_x1)
            crop_x2 = min(w, table_x2)
            crop_y1 = max(0, top_y1)
            crop_y2 = min(h, bottom_y2)
            
            # 截取图像
            cropped_img = rotated_img[crop_y1:crop_y2, crop_x1:crop_x2]
            logger.info(f"成功截取表格区域: 位置=({crop_x1},{crop_y1}), 尺寸={crop_x2-crop_x1}x{crop_y2-crop_y1}")
            
            # 将截取后的图像转换为可流式传输的格式
            is_success, buffer = cv2.imencode(".jpg", cropped_img)
            if not is_success:
                logger.error("无法将截取的图像编码为JPEG格式")
                raise Exception("无法将截取的图像编码为JPEG格式")
            
            # 保存截取后的图像（可选）
            os.makedirs("./output", exist_ok=True)
            cropped_img_path = f"./output/{timestamp}.jpg"
            cv2.imwrite(cropped_img_path, cropped_img)
            logger.info(f"已截取区域并保存到: {cropped_img_path}")
            
            # 返回截取后的图像
            return StreamingResponse(
                io.BytesIO(buffer.tobytes()),
                media_type="image/jpeg"
            )
        else:
            # 未找到表格区域或文本区域，返回原图
            logger.warning("未找到表格区域或文本区域，将返回原图")
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
        # 处理过程中发生错误，返回原图
        logger.error(f"表格处理过程中出错: {str(e)}，将返回原图")
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