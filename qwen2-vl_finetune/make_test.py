# 导入库
import os
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm  # 用于显示进度条

# 加载验证集数据
def load_captions(caption_file):
    captions = {}
    with open(caption_file, "r") as f:
        for line in f:
            # image_path, caption = line.strip().split(",", 1)
            image_path=line.strip()
            captions[image_path] = "_"
    return captions

# 直接加载模型和处理器
model_path = "fine-tuned-lora-model"  # 替换为模型路径
model = AutoModelForVision2Seq.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# # 加载验证集
caption_file = "data/captions_test.txt"  # 替换为captions.txt路径
captions = load_captions(caption_file)


# 平滑函数（用于处理短句子）
smoother = SmoothingFunction()

# 调整图像尺寸
target_size = (280, 280)  # 你可以根据需要调整尺寸

with open("test.txt","w") as file:
# 遍历验证集中的每张图片
    for image_path,_ in tqdm(captions.items()):


        # 加载图像
        full_image_path = os.path.join("data", image_path)  # 将相对路径与 data 文件夹拼接
        image = Image.open(full_image_path).convert("RGB")
        
        image = image.resize(target_size)

        # 构造输入消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},  # 图像路径
                    {"type": "text", "text": "describe this image"},  # 提示文本
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
        generated_caption = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        file.write( image_path+","+generated_caption+"\n")
        print(generated_caption)
