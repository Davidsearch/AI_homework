# 导入库
import torch
from torch.utils.data import Dataset, DataLoader
from modelscope import snapshot_download, AutoTokenizer
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, DataCollatorForSeq2Seq
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model  # 使用 LoraConfig
from PIL import Image
import os
import deepspeed
from tqdm import tqdm

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

# DeepSpeed 配置
deepspeed_config = {
    "train_batch_size": 8,  # 与 DataLoader 的 batch_size 一致
    "gradient_accumulation_steps": 4,  # 梯度累积步数
    "fp16": {
        "enabled": False,  # 禁用 FP16
    },
    "zero_optimization": {
        "stage": 2,  # 优化显存
        "contiguous_gradients": True, #减少反向传播期间的内存碎片
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8,
    },
    # "zero_optimization": {
    #     "stage": 3,
    #     "contiguous_gradients": True,
    #     "stage3_max_live_parameters": 1e9,
    #     "stage3_max_reuse_distance": 1e9,
    #     "stage3_prefetch_bucket_size": 1e7,
    #     "stage3_param_persistence_threshold": 1e5,
    #     "reduce_bucket_size": 1e7,
    #     "sub_group_size": 1e9,
    #     "offload_optimizer": {
    #         "device": "cpu"
    #      },
    #     "offload_param": {
    #         "device": "cpu"
    #    }
    # },
    # "zero_optimization": {
    #     "stage": 3,  # 使用 ZeRO 阶段 3
    #     "contiguous_gradients": True,  # 减少反向传播期间的内存碎片
    #     "stage3_max_live_parameters": 1e9,  # 最大活跃参数数量
    #     "stage3_max_reuse_distance": 1e9,  # 最大重用距离
    #     "stage3_prefetch_bucket_size": 1e7,  # 预取桶大小
    #     "stage3_param_persistence_threshold": 1e5,  # 参数持久化阈值
    #     "reduce_bucket_size": 1e7,  # 减少桶大小
    #     "sub_group_size": 1e9,  # 子组大小
    #     "offload_optimizer": {
    #         "device": "cpu",  # 将优化器状态卸载到 CPU
    #         "pin_memory": True  # 使用锁页内存，加速数据传输
    #     },
    #     "offload_param": {
    #         "device": "cpu",  # 将模型参数卸载到 CPU
    #         "pin_memory": True  # 使用锁页内存，加速数据传输
    #     }
    # },
    "optimizer": {
        "type": "Adam",  # 使用 DeepSpeed 支持的优化器类型
        "params": {
            "lr": 5e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
    },
}

# 初始化 DeepSpeed
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=deepspeed_config,
)

# 检查模型参数的设备
print("Checking model parameters device:")
for name, param in model.named_parameters():
    print(f"{name}: device={param.device}, requires_grad={param.requires_grad}, grad={param.grad}")

# 检查优化器类型
print("Optimizer type:", type(optimizer))

# 检查优化器状态的设备
if hasattr(optimizer, 'optimizer'):
    print("Optimizer is wrapped by DeepSpeed. Checking internal optimizer:")
    internal_optimizer = optimizer.optimizer
    if hasattr(internal_optimizer, 'param_groups'):
        print("Checking internal optimizer parameters device:")
        for param_group in internal_optimizer.param_groups:
            if 'params' in param_group:
                for param in param_group['params']:
                    print(f"Optimizer param: {param.device}")
            else:
                print("No 'params' key in param_group")
    else:
        print("Internal optimizer does not have 'param_groups' attribute")
else:
    print("Optimizer is not wrapped by DeepSpeed")

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
        device_gpu = 0
        batch = {k: v.to(device_gpu) for k, v in batch.items()}
        
        # 前向传播
        outputs = model(**batch)
        
        loss = outputs.loss

        # 反向传播和梯度累积
        model.backward(loss)
        model.step()

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