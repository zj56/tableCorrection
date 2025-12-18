'''
Author: jzousz zj56@qq.com
Date: 2025-09-26 13:53:09
LastEditors: jzousz zj56@qq.com
LastEditTime: 2025-09-28 09:19:46
FilePath: \paddle\test_table_service.py
'''
import requests
import argparse
from pathlib import Path
import os
import os
# from dashscope import MultiModalConversation
# import dashscope

# 定义根目录路径
def get_root():
    """获取项目根目录"""
    return Path(__file__).parent

ROOT = get_root()


def test_service(image_path, server_url="http://localhost:8001", output_dir=None):
    """测试表格矫正服务"""
    endpoint = "/correct_detection_table"
    full_url = f"{server_url}{endpoint}"
    print(f"测试服务: {full_url}")
    print(f"测试图片: {image_path}")

    # 检查图片是否存在
    if not Path(image_path).exists():
        print(f"❌ 错误: 图片文件不存在: {image_path}")
        return None

    with open(image_path, 'rb') as f:
        files = {'file': (Path(image_path).name, f, 'image/jpeg')}

        try:
            response = requests.post(full_url, files=files)
            print(f"响应状态码: {response.status_code}")

            if response.status_code == 200:
                # 使用指定的输出目录或默认目录
                if output_dir is None:
                    output_dir = ROOT / "output_test"
                else:
                    output_dir = Path(output_dir)

                # 创建输出目录（如果不存在）
                output_dir.mkdir(parents=True, exist_ok=True)

                # 保存矫正后的图片
                output_filename = f"corrected_{Path(image_path).name}"
                output_path = str(output_dir / output_filename)

                with open(output_path, 'wb') as out_f:
                    out_f.write(response.content)

                print(f"✅ 处理成功!")
                print(f"📁 结果保存到: {output_path}")
                return output_path
            else:
                print(f"❌ 处理失败: {response.status_code}")
                # 如果是JSON响应，尝试打印错误详情
                try:
                    error_detail = response.json().get('detail', '')
                    print(f"错误详情: {error_detail}")
                except:
                    print(f"响应内容: {response.text}")
                return None

        except requests.exceptions.ConnectionError:
            print(f"❌ 无法连接到服务: {server_url}")
            print("请确保服务已启动: python table_correction_service.py")
            return None
        except Exception as e:
            print(f"❌ 请求失败: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description='测试表格矫正服务')
    # parser.add_argument('--image', default=ROOT / "23.jpg", type=str, help='测试图片路径')
    parser.add_argument('--folder', type=str,default=ROOT / "lin", help='测试图片文件夹路径（批量处理）')
    parser.add_argument('--server', type=str, default='http://localhost:8001', help='服务器地址')

    args = parser.parse_args()

    # # 检查图片是否存在
    # if not Path(args.image).exists():
    #     print(f"❌ 错误: 图片文件不存在: {args.image}")
    #     # 尝试使用示例图片路径
    #     example_image = ROOT / "23.jpg"
    #     if example_image.exists():
    #         print(f"尝试使用示例图片: {example_image}")
    #         args.image = str(example_image)
    #     else:
    #         return 1

    # 如果指定了文件夹，则批量处理两层目录结构
    if hasattr(args, 'folder') and args.folder:
        root_folder = Path(args.folder)
        if not root_folder.exists() or not root_folder.is_dir():
            print(f"❌ 错误: 文件夹不存在或不是有效的目录: {args.folder}")
            return 1

        # 支持的图片格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

        # 统计信息
        total_success = 0
        total_failed = 0

        # 创建总输出目录
        output_root = ROOT / "output_test"
        output_root.mkdir(parents=True, exist_ok=True)

        print(f"\n开始批量处理两层目录结构...\n")

        # 第一层遍历：遍历所有子文件夹
        for subfolder in sorted(root_folder.iterdir()):
            if not subfolder.is_dir():
                continue

            # 获取子文件夹中的所有图片文件
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(subfolder.glob(f'*{ext}')))
                image_files.extend(list(subfolder.glob(f'*{ext.upper()}')))

            if not image_files:
                print(f"⚠️ 警告: 在子文件夹 {subfolder.name} 中未找到支持的图片文件")
                continue

            print(f"\n处理子文件夹: {subfolder.name}")
            print(f"找到 {len(image_files)} 个图片文件")

            # 为当前子文件夹创建对应的结果文件夹
            sub_output_dir = output_root / subfolder.name
            sub_output_dir.mkdir(parents=True, exist_ok=True)

            # 统计信息
            success_count = 0
            failed_count = 0

            # 逐个处理图片
            for i, image_path in enumerate(image_files, 1):
                # file_path = f"file://{image_path}"
                # messages = [
                print(f"处理图片 {i}/{len(image_files)}: {image_path.name}")
                print("=" * 50)

                # 调用处理函数，并传递输出目录
                result = test_service(str(image_path), args.server, str(sub_output_dir))
                print("=" * 50)

                if result:
                    success_count += 1
                    print(f"✅ 图片 {image_path.name} 处理成功!")
                else:
                    failed_count += 1
                    print(f"❌ 图片 {image_path.name} 处理失败!")
                print()

            # 更新总统计
            total_success += success_count
            total_failed += failed_count

            print(f"\n子文件夹 {subfolder.name} 处理完成:")
            print(f"成功数: {success_count}")
            print(f"失败数: {failed_count}")
            print("=" * 50)

        # 打印总统计信息
        print("=" * 50)
        print(f"所有子文件夹处理完成!")
        print(f"总成功数: {total_success}")
        print(f"总失败数: {total_failed}")
        print("=" * 50)

        return 0 if total_success > 0 else 1
    else:
        # 单文件处理模式
        print("=" * 50)
        result = test_service(args.image, args.server)
        print("=" * 50)

    if result:
        print("✨ 测试完成!")
        return 0
    else:
        print("💥 测试失败!")
        return 1


if __name__ == "__main__":
    exit(main())
