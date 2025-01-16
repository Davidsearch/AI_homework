import os
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# 设置路径
current_dir = os.getcwd()  # 当前目录
model_dir = os.path.join(current_dir, "checkpoint_epoch_5")  # 模型保存目录
test_image_path = os.path.join(current_dir, "flow_chart_of_image_captioning.png")  # 测试图片路径

# 加载训练后的模型和分词器
processor = BlipProcessor.from_pretrained(model_dir)
model = BlipForConditionalGeneration.from_pretrained(model_dir).to("cuda")

# 测试图片生成描述
def generate_caption(image_path):
    """
    生成图片的描述
    :param image_path: 图片路径
    :return: 生成的描述文本
    """
    # 加载图片
    image = Image.open(image_path).convert("RGB")
    
    # 使用处理器预处理图片
    inputs = processor(image, return_tensors="pt").to("cuda")
    
    # 使用模型生成描述
    with torch.no_grad():
        out = model.generate(**inputs, max_length=50)  # 设置 max_length
    
    # 解码生成的描述
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# 测试一张图片
if __name__ == "__main__":
    if not os.path.exists(test_image_path):
        print(f"测试图片不存在: {test_image_path}")
    else:
        caption = generate_caption(test_image_path)
        print(f"生成的描述: {caption}")