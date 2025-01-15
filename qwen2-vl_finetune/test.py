# 导入库
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from PIL import Image

# 直接加载模型和处理器
# model_path = "Qwen2-VL"  # 替换为你的模型路径
model_path = "fine-tuned-lora-model"
model = AutoModelForVision2Seq.from_pretrained(model_path)


processor = AutoProcessor.from_pretrained(model_path)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载测试图像
image_path = "a.jpg"  # 替换为你的测试图像路径
image = Image.open(image_path).convert("RGB")  # 确保图像是 RGB 格式

# 调整图像尺寸
target_size = (1080, 1080)  # 你可以根据需要调整尺寸
image = image.resize(target_size)

# 构造输入消息
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},  # 图像路径
            {"type": "text", "text": "描述该图片"},  # 提示文本
        ],
    }
]

# 使用 processor 处理输入
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(
    text=[text],
    images=[image],  # 直接传入 PIL 图像
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(device)

# 推理：生成文本描述
model.eval()
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)

# 解码生成的文本
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

# 打印生成的文本
print("Generated Text:", output_text[0])