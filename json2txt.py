import os
import json
import argparse
import shutil
from PIL import Image

def convert_labelme_to_yolo(labelme_json_path, image_width, image_height, class_mapping):
    """将单个 Labelme JSON 文件转换为 YOLO 格式的 TXT 文件"""
    yolo_lines = []
    try:
        with open(labelme_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']
            if label not in class_mapping:
                print(f"警告: 未定义的类别 '{label}'，跳过此标注")
                continue
            class_id = class_mapping[label]
            if shape['shape_type'] == 'rectangle':
                if len(points) != 2:
                    print(f"警告: 矩形标注应有 2 个点，但找到 {len(points)} 个，跳过")
                    continue
                x1, y1 = points[0]
                x2, y2 = points[1]
                x_center = (x1 + x2) / 2 / image_width
                y_center = (y1 + y2) / 2 / image_height
                width = abs(x2 - x1) / image_width
                height = abs(y2 - y1) / image_height
            elif shape['shape_type'] == 'polygon':
                if len(points) < 3:
                    print(f"警告: 多边形标注至少需要 3 个点，但找到 {len(points)} 个，跳过")
                    continue
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x1 = min(x_coords)
                y1 = min(y_coords)
                x2 = max(x_coords)
                y2 = max(y_coords)
                x_center = (x1 + x2) / 2 / image_width
                y_center = (y1 + y2) / 2 / image_height
                width = (x2 - x1) / image_width
                height = (y2 - y1) / image_height
            else:
                print(f"警告: 不支持的标注类型 '{shape['shape_type']}'，跳过")
                continue
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    except Exception as e:
        print(f"错误: 处理文件 {labelme_json_path} 时出错: {e}")
    return yolo_lines

def extract_classes_from_json(json_dir):
    """从 JSON 文件中提取所有类别名并生成映射"""
    classes = set()
    for root, dirs, files in os.walk(json_dir):
        for filename in files:
            if filename.endswith('.json'):
                json_path = os.path.join(root, filename)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for shape in data.get('shapes', []):
                            classes.add(shape['label'])
                except Exception as e:
                    print(f"警告: 无法读取文件 {filename}: {e}")
    class_list = sorted(list(classes))
    class_mapping = {cls: i for i, cls in enumerate(class_list)}
    return class_mapping, class_list

def find_image_file(json_basename, image_dir, extensions=['.jpg', '.jpeg', '.png']):
    """查找与 JSON 文件对应的图像文件，支持多种扩展名"""
    for ext in extensions:
        image_path = os.path.join(image_dir, json_basename + ext)
        if os.path.exists(image_path):
            return image_path, ext
    return None, None

def process_directory(input_dir, output_root, image_source_dir, class_mapping, copy_images):
    """
    处理目录中的所有 Labelme JSON 文件
    :param input_dir: Labelme JSON 文件所在目录
    :param output_root: 输出根目录（train_data）
    :param image_source_dir: 原始图像所在目录（train_data/images 上层或其他位置 ）
    :param class_mapping: 类别映射字典
    :param copy_images: 是否复制图像到指定目录
    """
    # 构建标注和图像输出目录
    labels_output_dir = os.path.join(output_root, 'labels', 'train')
    images_output_dir = os.path.join(output_root, 'images', 'train')
    os.makedirs(labels_output_dir, exist_ok=True)
    if copy_images:
        os.makedirs(images_output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        for json_file in files:
            if json_file.endswith('.json'):
                json_path = os.path.join(root, json_file)
                json_basename = os.path.splitext(json_file)[0]
                image_source_path = os.path.join(root)
                # 查找对应图像
                image_path, image_ext = find_image_file(json_basename, image_source_path)
                if image_path is None:
                    print(f"警告: 找不到与 {json_file} 对应的图像文件，跳过")
                    continue
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                except Exception as e:
                    print(f"错误: 无法获取图像 {image_path} 的尺寸: {e}")
                    continue
                # 转换标注
                yolo_lines = convert_labelme_to_yolo(json_path, width, height, class_mapping)
                if yolo_lines:
                    txt_file = json_basename + '.txt'
                    txt_path = os.path.join(labels_output_dir, txt_file)
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(yolo_lines))
                    # 复制图像
                    if copy_images:
                        target_image_path = os.path.join(images_output_dir, json_basename + image_ext)
                        shutil.copy2(image_path, target_image_path)
                        print(f"已转换并复制: {json_file} -> {txt_file} (图像: {os.path.basename(image_path)})")
                    else:
                        print(f"已转换: {json_file} -> {txt_file}")
                else:
                    print(f"警告: {json_file} 没有有效标注，未生成 TXT 文件")

def main():
    parser = argparse.ArgumentParser(description='将 Labelme 标注转换为 YOLO 格式')
    parser.add_argument('--input', default='./origin_data/', help='包含 Labelme JSON 文件的目录')
    parser.add_argument('--output-root', default='./train_data', help='输出根目录（train_data）')
    parser.add_argument('--image-source', default='./origin_data/', help='原始图像所在目录')
    parser.add_argument('--save-classes', default='./train_data/labels/train/classes.txt', help='保存类别映射到文件')
    parser.add_argument('--copy-images', default='./train_data', help='将对应的图像复制到输出目录')
    args = parser.parse_args()
    # 提取类别
    class_mapping, class_list = extract_classes_from_json(args.input)
    print(f"已从 JSON 文件中提取 {len(class_mapping)} 个类别: {', '.join(class_list)}")
    # 保存类别文件
    if args.save_classes:
        with open(args.save_classes, 'w', encoding='utf-8') as f:
            f.write('\n'.join(class_list))
        print(f"类别映射已保存到: {args.save_classes}")
    # 处理目录
    process_directory(args.input, args.output_root, args.image_source, class_mapping, args.copy_images)
    print("\n处理完成!")
    if args.copy_images:
        print(f"标注文件保存在: {os.path.join(args.output_root, 'labels', 'train')}")
        print(f"图像保存在: {os.path.join(args.output_root, 'images', 'train')}")
    else:
        print(f"标注文件保存在: {os.path.join(args.output_root, 'labels', 'train')}")

if __name__ == "__main__":
    main()