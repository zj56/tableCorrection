# from paddleocr import TextDetection
# model = TextDetection(model_name="PP-OCRv5_server_det")
# output = model.predict("21.jpg", batch_size=1)
# for res in output:
#     res.print()
#     res.save_to_img(save_path="./output/")
#     res.save_to_json(save_path="./output/res.json")

from paddleocr import DocImgOrientationClassification
import cv2
import numpy as np
from paddleocr import TextDetection
import os
import json
import requests
import argparse
from pathlib import Path
import io

# 确保output目录存在
output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"创建output目录: {output_dir}")
test_img = "23.jpg"
model = DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori")
output = model.predict(test_img,  batch_size=1)
rotated_img = None
for res in output:
    res.print(json_format=True)
    res.save_to_img("./output/demo.png")
    # 保存JSON文件
    json_file_path = "./output/res.json"
    res.save_to_json(json_file_path)
    
    # 尝试获取旋转角度信息 - 从保存的JSON文件中读取
    try:
        import json
        import os
        
        # 检查JSON文件是否存在
        if os.path.exists(json_file_path):
            # 读取JSON文件
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 从JSON数据中获取label_names
            if 'label_names' in data and data['label_names']:
                # 将label_names中的值转换为数字
                rotation_angle = int(data['label_names'][0])
                
                # 读取原始图像
                img = cv2.imread(test_img)
                if img is not None:
                    # 根据旋转角度执行相应的旋转操作
                    if rotation_angle == 90:
                        # 旋转90度（顺时针）
                        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    elif rotation_angle == 180:
                        # 旋转180度
                        rotated_img = cv2.rotate(img, cv2.ROTATE_180)
                    elif rotation_angle == 270:
                        # 旋转270度（顺时针，等同于逆时针旋转90度）
                        rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif rotation_angle != 0:
                        # 对于其他角度，使用通用旋转函数
                        (h, w) = img.shape[:2]
                        center = (w // 2, h // 2)
                        # 获取旋转矩阵
                        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                        # 执行旋转
                        rotated_img = cv2.warpAffine(img, M, (w, h))
                    else:
                        # 0度不需要旋转
                        print("旋转角度为0，无需旋转图像")
                        rotated_img = img.copy()
                    
                    # 保存旋转后的图像
                    save_path = f"./output/rotated_{rotation_angle}_96.jpg"
                    cv2.imwrite(save_path, rotated_img)
                    print(f"图像已旋转{rotation_angle}度并保存到: {save_path}")
                else:
                    print("无法读取图像文件")
            else:
                print("JSON文件中未找到有效的label_names信息")
        else:
            print(f"JSON文件不存在: {json_file_path}")
    except ValueError as ve:
        print(f"无法将label_names值转换为数字: {ve}")
    except json.JSONDecodeError as je:
        print(f"JSON文件解析错误: {je}")
    except Exception as e:
        print(f"处理旋转时出错: {e}")

server_url="http://127.0.0.1:8000"
import requests
import argparse
from pathlib import Path
import io
masked_img = None
# 检查rotated_img是否存在且有效
if rotated_img is not None:
    try:
        # 使用max_rectangle_detector.py中的方法获取最大表格区域
        from max_rectangle_detector import MaxRectangleDetector
        import cv2  # 确保导入cv2
        
        # 初始化最大矩形检测器
        detector = MaxRectangleDetector()
        
        # 获取最大矩形框（模拟表格区域）
        # get_max_rectangle返回的是(处理后的图像, 最大矩形框顶点)的元组
        result = detector.get_max_rectangle(rotated_img)
        
        if result is not None:
            result_image, rect_points = result
            
            # 创建rotated_img的副本进行mask处理
            masked_img = rotated_img.copy()
            
            # 计算矩形的边界框坐标
            # 从rect_points中提取x和y坐标
            x_coords = rect_points[:, 0]
            y_coords = rect_points[:, 1]
            x1 = int(min(x_coords))
            y1 = int(min(y_coords))
            x2 = int(max(x_coords))
            y2 = int(max(y_coords))
            
            # 使用黑色矩形覆盖检测区域（最大表格区域）
            # cv2.rectangle参数说明：
            # 1. 图像对象
            # 2. 左上角坐标(x1, y1)
            # 3. 右下角坐标(x2, y2)
            # 4. 颜色(0, 0, 0)表示黑色
            # 5. 厚度-1表示填充整个矩形
            cv2.rectangle(masked_img, (x1, y1), (x2, y2), (0, 0, 0), -1)
            
            # 保存mask后的图像
            masked_img_path = "./output/masked_rotated_image.jpg"
            cv2.imwrite(masked_img_path, masked_img)
            print(f"已对最大表格区域进行mask处理并保存到: {masked_img_path}")
            
            # 保存检测到的表格区域坐标，用于后续查找表格标题
            table_regions = []
            
            # 将最大表格区域加入到table_regions
            max_table_region = {
                'bbox': (x1, y1, x2, y2),
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'width': x2 - x1,
                'height': y2 - y1
            }
            table_regions.append(max_table_region)
            
            print(f"已将最大表格区域加入table_regions: {max_table_region}")
        else:
            print("未检测到有效的表格区域")
            masked_img = rotated_img.copy()
            # 创建空的table_regions
            table_regions = []
    except ImportError:
        print("无法导入MaxRectangleDetector，请确保max_rectangle_detector.py文件存在")
        masked_img = rotated_img.copy()
        # 创建空的table_regions
        table_regions = []
    except Exception as e:
        print(f"获取最大表格区域时出错: {e}")
        masked_img = rotated_img.copy()
        # 创建空的table_regions
        table_regions = []




model_text = TextDetection(model_name="PP-OCRv5_server_det")
output_text = model_text.predict(masked_img, batch_size=1)

# 存储所有文本区域信息
text_regions = []
print("文本检测结果:")
for res in output_text:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res_text.json")
    # print(res.json())
    
    # 从JSON文件读取文本区域信息
    try:
        json_file_path = "./output/res_text.json"
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'dt_polys' in data and 'dt_scores' in data:
                dt_polys = data['dt_polys']
                dt_scores = data['dt_scores']
                
                # 遍历所有检测到的文本框
                for i, poly in enumerate(dt_polys):
                    # 计算文本框的边界框
                    x_coords = [point[0] for point in poly]
                    y_coords = [point[1] for point in poly]
                    x1, y1 = int(min(x_coords)), int(min(y_coords))
                    x2, y2 = int(max(x_coords)), int(max(y_coords))
                    
                    text_regions.append({
                        'bbox': (x1, y1, x2, y2),
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                        'score': dt_scores[i]
                    })
                print(f"从JSON文件成功读取{len(text_regions)}个文本区域信息")
            else:
                print("JSON文件中缺少dt_polys或dt_scores字段")
        else:
            print(f"JSON文件不存在: {json_file_path}")
    except json.JSONDecodeError as je:
        print(f"JSON文件解析错误: {je}")
    except Exception as e:
        print(f"读取JSON文件时出错: {e}")
print(text_regions)
# 查找表格标题
if 'table_regions' in locals() and table_regions and text_regions:
    table_titles = []
    
    # 检查text_regions中心点是否在table_regions的两条垂直线区域内
    print("\n=== 文本区域中心点检查结果 ===")
    for table_idx, table_region in enumerate(table_regions):
        table_x1, table_y1, table_x2, table_y2 = table_region['bbox']
        print(f"\n表格 {table_idx+1} (x范围: {table_x1}-{table_x2}):")
        
        # 收集符合条件的文本区域
        matched_text_regions = []
        
        for text_idx, text_region in enumerate(text_regions):
            text_center_x, text_center_y = text_region['center']
            
            # 检查文本区域的x中心点是否在表格区域的x1和x2之间
            if table_x1 <= text_center_x <= table_x2:
                matched_text_regions.append({
                    'text_idx': text_idx+1,
                    'center': (text_center_x, text_center_y),
                    'bbox': text_region['bbox'],
                    'score': text_region['score']
                })
        
        # 打印符合条件的文本区域
        if matched_text_regions:
            print(f"  找到 {len(matched_text_regions)} 个文本区域的中心点在表格的垂直线区域内:")
            for matched_text in matched_text_regions:
                print(f"    文本区 {matched_text['text_idx']}: 中心点({matched_text['center'][0]}, {matched_text['center'][1]}), 置信度: {matched_text['score']:.2f}")
        else:
            print("  未找到中心点在表格垂直线区域内的文本区域")
        
        # 在图像上标注符合条件的文本区域
        if matched_text_regions and 'rotated_img' in locals() and rotated_img is not None:
            # 创建图像副本用于标注
            annotated_img = rotated_img.copy()
            
            # 绘制表格区域的两条垂直线
            cv2.line(annotated_img, (table_x1, 0), (table_x1, annotated_img.shape[0]), (0, 0, 255), 2)
            cv2.line(annotated_img, (table_x2, 0), (table_x2, annotated_img.shape[0]), (0, 0, 255), 2)
            
            # 标注符合条件的文本区域
            for matched_text in matched_text_regions:
                text_idx = matched_text['text_idx']
                x1, y1, x2, y2 = matched_text['bbox']
                center_x, center_y = matched_text['center']
                
                # 绘制文本区域的边界框（绿色）
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 标注文本区域编号和中心点
                label = f"文本区{text_idx}"
                cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(annotated_img, (center_x, center_y), 3, (255, 0, 0), -1)
            
            # 保存标注后的图像
            annotated_img_path = f"./output/annotated_matched_text_regions_table_{table_idx+1}.jpg"
            cv2.imwrite(annotated_img_path, annotated_img)
            print(f"已在图像上标注符合条件的文本区域并保存到: {annotated_img_path}")

            # 截取区域：横坐标为表格的x1和x2之间，纵坐标根据检测到的文本区域确定
            # 确定纵坐标范围
            if matched_text_regions:
                # 有检测到文本区域，使用最上方文本框的y1和最下方文本框的y2
                top_y1 = min([text['bbox'][1] for text in matched_text_regions])
                bottom_y2 = max([text['bbox'][3] for text in matched_text_regions])
                print(f"  基于文本区域确定纵坐标范围: {top_y1}-{bottom_y2}")
            else:
                # 没有检测到文本框，使用表格区域的y1和y2
                top_y1 = table_y1
                bottom_y2 = table_y2
                print(f"  没有检测到文本区域，使用表格区域纵坐标范围: {top_y1}-{bottom_y2}")
            
            # 截取区域
            # 确保坐标在图像范围内
            h, w = rotated_img.shape[:2]
            crop_x1 = max(0, table_x1)
            crop_x2 = min(w, table_x2)
            crop_y1 = max(0, top_y1)
            crop_y2 = min(h, bottom_y2)
            
            # 截取图像
            cropped_img = rotated_img[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # 保存截取后的图像
            cropped_img_path = f"./output/cropped_region_table_{table_idx+1}.jpg"
            cv2.imwrite(cropped_img_path, cropped_img)
            print(f"  已截取区域并保存到: {cropped_img_path}")
            print(f"  截取区域尺寸: 宽度={crop_x2-crop_x1}, 高度={crop_y2-crop_y1}")
    

