# 导入库
import torch
from torch.utils.data import Dataset, DataLoader
from modelscope import snapshot_download, AutoTokenizer
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, DataCollatorForSeq2Seq
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model  # 使用普通的 LoRA
from PIL import Image
import os
from tqdm import tqdm
from torch.optim import AdamW  # 使用 PyTorch 自带的优化器

# 定义 LoRA 配置
lora_config = LoraConfig(
    task_type="CAUSAL_LM",  # 任务类型
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 目标模块
    r=8,  # LoRA 的秩
    lora_alpha=32,  # 缩放因子
    lora_dropout=0.1,  # Dropout 比例
    bias="none",  # 是否训练偏置项
)

# 定义加载模型的路径
load_from_pretrained = True  # 设置为 True 从预训练模型加载，False 从微调模型加载
pretrained_model_name = "Qwen2-VL"  # 预训练模型名称
fine_tuned_model_path = "./fine-tuned-lora-model_0108"  # 微调后的模型路径

# 分支逻辑：加载预训练模型或微调后的模型
if load_from_pretrained:
    # 从预训练模型加载
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        pretrained_model_name, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(pretrained_model_name)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, use_fast=False, trust_remote_code=True)
else:
    # 从微调后的模型加载
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        fine_tuned_model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(fine_tuned_model_path)
    tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path, use_fast=False, trust_remote_code=True)

# 启用输入梯度
model.enable_input_require_grads()

# 打印调试信息
print("Model loaded from:", pretrained_model_name if load_from_pretrained else fine_tuned_model_path)
print("Model:", model)
print("LoRA Config:", lora_config)

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
                {"type": "text", "text": "描述该图片,先用知识库中的英文生成但不输出，然后翻译成中文输出"},  # 提示文本
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
num_epochs = 1

# 创建数据集实例
train_dataset = ImageTextDataset(data, processor)

# 创建 DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True))

# 定义优化器
optimizer = AdamW(model.parameters(), lr=5e-5)  # 使用 AdamW 优化器

# 训练循环
model.train()

# 检查模型参数所在的设备
print(f"Model parameters device: {next(model.parameters()).device}")

# 检查第一个 batch 的设备
sample_batch = next(iter(train_dataloader))
print(f"Sample batch device: {sample_batch['input_ids'].device}")

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Training Epoch {epoch+1}"):
        # 将数据移动到设备
        device = next(model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 前向传播
        outputs = model(**batch)
        loss = outputs.loss

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 打印损失
        epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}, Step {step + 1}, Loss: {loss.item()}")

    print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(train_dataloader):.4f}")

# 保存微调后的模型
model.save_pretrained("./fine-tuned-lora-model_0108_cont")
processor.save_pretrained("./fine-tuned-lora-model_0108_cont")

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
import os
from PIL import Image

# 指定文件夹路径
test_image_folder = "./data/images/"  # 替换为你的测试图片文件夹路径

# 遍历文件夹中的所有图片
for image_name in os.listdir(test_image_folder):
    # 检查文件是否为图片（根据扩展名过滤）
    if image_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
        # 构造完整图片路径
        image_path = os.path.join(test_image_folder, image_name)
        
        # 生成描述
        try:
            description = generate_description(image_path)
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            continue
        # 输出结果
        print(f"Image: {image_name}")
        print("Generated Description:", description)
        print("-" * 50)  # 分隔线