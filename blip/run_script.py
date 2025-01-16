import os
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# 设置路径
current_dir = os.getcwd()  # 当前目录
model_dir = os.path.join(current_dir, "blip-image-captioning-base")  # 训练后模型保存目录
image_folder = os.path.join(current_dir, "test_images")  # 测试图片文件夹路径

# 支持的图片格式
supported_formats = ['jpg', 'jpeg', 'png', 'bmp', 'gif']

# 加载训练后的模型和分词器
processor = BlipProcessor.from_pretrained(model_dir)
model = BlipForConditionalGeneration.from_pretrained(model_dir).to("cuda")

# 遍历文件夹中的所有图片
for image_name in os.listdir(image_folder):
    # 检查文件格式
    if image_name.split('.')[-1].lower() not in supported_formats:
        print(f"跳过不支持的文件格式: {image_name}")
        continue  # 跳过不支持的文件格式

    # 加载图片
    image_path = os.path.join(image_folder, image_name)
    try:
        raw_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"无法加载图片 {image_name}: {e}")
        continue

    # 条件图像描述生成
    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs)
    caption_conditional = processor.decode(out[0], skip_special_tokens=True)

    # 无条件图像描述生成
    inputs = processor(raw_image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs)
    caption_unconditional = processor.decode(out[0], skip_special_tokens=True)

    # 输出结果
    print(f"图片: {image_name}")
    print(f"条件描述: {caption_conditional}")
    print(f"无条件描述: {caption_unconditional}")
    print("-" * 50)