import os
import random
import shutil

# 数据集根目录
root_dir = "./train_data"
# 图片和标注的原始目录（train 目录）
image_train_dir = os.path.join(root_dir, "images", "train")
label_train_dir = os.path.join(root_dir, "labels", "train")
# 验证集图片和标注目标目录（val 目录）
image_val_dir = os.path.join(root_dir, "images", "val")
label_val_dir = os.path.join(root_dir, "labels", "val")

# 第一步：删除 /images/train/ 目录下没有标注的图片
def delete_unannotated_images():
    # 获取图片文件名（不含扩展名）集合
    image_names = set([os.path.splitext(img)[0] for img in os.listdir(image_train_dir)])
    # 获取标注文件名（不含扩展名）集合
    label_names = set([os.path.splitext(label)[0] for label in os.listdir(label_train_dir)])

    # 遍历图片，找出没有对应标注的图片并删除
    for img in os.listdir(image_train_dir):
        img_name = os.path.splitext(img)[0]
        if img_name not in label_names:
            img_path = os.path.join(image_train_dir, img)
            os.remove(img_path)
            print(f"已删除无标注图片: {img_path}")

# 第二步：划分验证集，将 10% 的图片及对应标注划分到验证集目录
def split_validation_set():
    # 创建验证集图片和标注目录（如果不存在）
    os.makedirs(image_val_dir, exist_ok=True)
    os.makedirs(label_val_dir, exist_ok=True)

    # 获取剩余图片文件名（不含扩展名）列表
    remaining_image_names = [os.path.splitext(img)[0] for img in os.listdir(image_train_dir)]
    # 计算验证集数量（取整）
    val_count = int(len(remaining_image_names) * 0.1)
    # 随机选择要划分到验证集的图片名
    val_image_names = random.sample(remaining_image_names, val_count)

    # 移动图片和标注到验证集目录
    for img_name in val_image_names:
        # 图片扩展名，这里简单处理，假设都是常见图片格式，可根据实际情况调整
        img_ext = ".jpg"
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            possible_img_path = os.path.join(image_train_dir, img_name + ext)
            if os.path.exists(possible_img_path):
                img_ext = ext
                break
        img_path = os.path.join(image_train_dir, img_name + img_ext)
        label_path = os.path.join(label_train_dir, img_name + ".txt")

        # 移动图片到验证集图片目录
        shutil.move(img_path, os.path.join(image_val_dir, img_name + img_ext))
        # 移动标注到验证集标注目录
        shutil.move(label_path, os.path.join(label_val_dir, img_name + ".txt"))
        print(f"已移动到验证集: {img_path}, {label_path}")

if __name__ == "__main__":
    # 执行第一步：删除无标注图片
    delete_unannotated_images()
    # 执行第二步：划分验证集
    split_validation_set()
    print("数据集处理完成！")