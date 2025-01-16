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
            image_path, caption = line.strip().split(",", 1)
            captions[image_path] = caption
    return captions

# 直接加载模型和处理器
model_path = "Qwen2-VL"  # 替换为模型路径
model = AutoModelForVision2Seq.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载验证集
caption_file = "data/captions_val.txt"  # 替换为captions.txt路径
captions = load_captions(caption_file)

# 初始化 BLEU 分数
bleu_1_scores = []
bleu_2_scores = []
bleu_3_scores = []
bleu_4_scores = []

# 平滑函数（用于处理短句子）
smoother = SmoothingFunction()

# 调整图像尺寸
target_size = (280, 280)  # 你可以根据需要调整尺寸

i=1
# 遍历验证集中的每张图片
for image_path, true_caption in tqdm(captions.items()):
    if i==500:
        break
    i=i+1
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
    print(generated_caption)

    # 计算 BLEU 分数
    true_caption_tokens = true_caption.split()  # 将真实描述分词
    generated_caption_tokens = generated_caption.split()  # 将生成描述分词

    # 计算 BLEU-1, BLEU-2, BLEU-3, BLEU-4
    bleu_1 = sentence_bleu([true_caption_tokens], generated_caption_tokens, weights=(1, 0, 0, 0), smoothing_function=smoother.method1)
    bleu_2 = sentence_bleu([true_caption_tokens], generated_caption_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother.method1)
    bleu_3 = sentence_bleu([true_caption_tokens], generated_caption_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoother.method1)
    bleu_4 = sentence_bleu([true_caption_tokens], generated_caption_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother.method1)

    # 记录分数
    bleu_1_scores.append(bleu_1)
    bleu_2_scores.append(bleu_2)
    bleu_3_scores.append(bleu_3)
    bleu_4_scores.append(bleu_4)

    # 释放显存
    del inputs, generated_ids, generated_caption
    torch.cuda.empty_cache()

avg_bleu_1 = sum(bleu_1_scores) / 500
avg_bleu_2 = sum(bleu_2_scores) / 500
avg_bleu_3 = sum(bleu_3_scores) / 500
avg_bleu_4 = sum(bleu_4_scores) / 500
# # 计算平均 BLEU 分数
# avg_bleu_1 = sum(bleu_1_scores) / len(bleu_1_scores)
# avg_bleu_2 = sum(bleu_2_scores) / len(bleu_2_scores)
# avg_bleu_3 = sum(bleu_3_scores) / len(bleu_3_scores)
# avg_bleu_4 = sum(bleu_4_scores) / len(bleu_4_scores)

# 打印结果
print(f"BLEU-1: {avg_bleu_1:.4f}")
print(f"BLEU-2: {avg_bleu_2:.4f}")
print(f"BLEU-3: {avg_bleu_3:.4f}")
print(f"BLEU-4: {avg_bleu_4:.4f}")