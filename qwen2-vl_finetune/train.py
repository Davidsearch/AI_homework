# 导入库
import torch
from torch.utils.data import Dataset, DataLoader
from modelscope import snapshot_download, AutoTokenizer
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, DataCollatorForSeq2Seq
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from PIL import Image
import os

# 定义 LoRA 配置
lora_config = LoraConfig(
    task_type="CAUSAL_LM",  # 任务类型
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 目标模块
    r=64,  # LoRA 秩
    lora_alpha=16,  # 缩放因子
    lora_dropout=0.05,  # Dropout 比例
    bias="none",  # 是否训练偏置项
)

# 加载预训练模型和处理器
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen2-VL", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen2-VL")
tokenizer = AutoTokenizer.from_pretrained("Qwen2-VL", use_fast=False, trust_remote_code=True)
model.enable_input_require_grads()

# 将 LoRA 应用到模型
model = get_peft_model(model, lora_config)

# 打印可训练参数数量
model.print_trainable_parameters()

# 加载数据集
def load_data(data_dir):
    captions_path = os.path.join(data_dir, "captions.txt")
    with open(captions_path, "r") as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        image_path, text = line.strip().split(",", 1)
        data.append({"image_path": os.path.join(data_dir, image_path), "text": text})
    return data

def preprocess_function(examples):
    """
    数据预处理函数
    """
    MAX_LENGTH = 8192  # 最大序列长度
    output_content = examples["text"]
    
    # 构造输入消息
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": examples["image_path"],
                    "resized_height": 280,
                    "resized_width": 280,
                },
                {"type": "text", "text": "描述该图片"},  # 提示文本
            ],
        }
    ]
    
    # 使用 processor 处理数据
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)  # 获取图像数据（预处理过）
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # 将 tensor 转换为 list，方便拼接
    inputs = {key: value.tolist() for key, value in inputs.items()}
    instruction = inputs
    
    # 处理 response
    response = tokenizer(examples["text"], add_special_tokens=False)
    
    # 拼接 input_ids
    input_ids = instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    
    # 拼接 attention_mask
    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    
    # 生成 labels
    labels = [-100] * len(instruction["input_ids"][0]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    # 截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    # 转换为 tensor
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  # 由 (1, h, w) 变换为 (h, w)
    
    # 返回结果
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": inputs['pixel_values'],
        "image_grid_thw": inputs['image_grid_thw']
    }

# 创建数据集类
class ImageTextDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = preprocess_function(item)
        inputs = {k: v.squeeze() for k, v in inputs.items()}  # 去除 batch 维度
        return inputs

# 加载数据
data_dir = "data"  # 替换为你的数据文件夹路径
data = load_data(data_dir)

# 创建数据集实例
train_dataset = ImageTextDataset(data, processor)

# 创建 DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)) #batch_size=8 

# 设置训练参数
from transformers import AdamW, get_scheduler

optimizer = AdamW(model.parameters(), lr=1e-4)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# 训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练循环
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # 将数据移动到设备
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 前向传播
        outputs = model(**batch)
        loss = outputs.loss
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        # 打印损失
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# 保存微调后的模型
model.save_pretrained("./fine-tuned-lora-model")
processor.save_pretrained("./fine-tuned-lora-model")

# 推理：使用留出的图片测试模型
def generate_description(image_path, prompt="Describe the image:"):
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    
    # 构造输入消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},  # 图像路径
                {"type": "text", "text": prompt},  # 提示文本
            ],
        }
    ]
    
    # 使用 processor 处理输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    
    # 生成文本描述
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
    return output_text[0]

# 测试留出的图片
test_image_path = "./data/images/test_image.jpg"  # 替换为你的测试图片路径
description = generate_description(test_image_path)
print("Generated Description:", description)