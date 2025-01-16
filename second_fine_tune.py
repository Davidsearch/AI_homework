import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AdamW
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from pycocoevalcap.spice.spice import Spice
import random

# 设置训练轮数(全局变量)
num_epochs = 50  # 假设训练 50 个 epoch

# 设置路径
current_dir = os.getcwd()  # 当前目录（blip 目录）
parent_dir = os.path.dirname(current_dir)  # 上一级目录
data_dir = os.path.join(parent_dir,  'qwen2-vl_finetune', 'data')  # 数据集目录
model_dir = os.path.join(current_dir, 'checkpoint_epoch_5')  # 模型目录

# 加载 BLIP 模型和处理器
processor = BlipProcessor.from_pretrained(model_dir)
model = BlipForConditionalGeneration.from_pretrained(model_dir).to("cuda")

# 添加新 token 并调整模型的词嵌入层
additional_tokens = [" ", "[NEW_TOKEN_2]"]
processor.tokenizer.add_tokens(additional_tokens)
# 确保模型的词嵌入层与分词器的词汇表大小一致。这是非常重要的一步，能够避免由于词汇表不匹配导致的模型性能问题。
model.resize_token_embeddings(len(processor.tokenizer))

# 检查词汇表大小
print("扩展后的分词器词汇表大小:", processor.tokenizer.vocab_size)

# 加载数据集
def load_data(data_dir):
    captions_path = os.path.join(data_dir, "../","test.txt")
    with open(captions_path, "r") as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        image_path, text = line.strip().split(",", 1)
        data.append({"image_path": os.path.join(data_dir, image_path), "text": text})
    return data

# 数据集类
class ImageCaptionDataset(Dataset):
    def __init__(self, data, processor, transform=None):
        self.data = data
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image_path"]
        text = item["text"]
        
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, text
        except Exception as e:
            print(f"加载图像 {image_path} 失败: {e}")
            return None, None

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量，范围 [0, 1]
])

# 加载数据
data = load_data(data_dir)
dataset = ImageCaptionDataset(data, processor, transform=transform)

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Collate 函数
def collate_fn(batch):
    batch = [(img, text) for img, text in batch if img is not None and text is not None]
    if not batch:
        return None, None, None
    images, texts = zip(*batch)
    images = torch.stack(images, dim=0)
    inputs = processor(
        text=list(texts),
        images=list(images),
        return_tensors="pt",
        padding=True,
        truncation=True,
        do_rescale=False  # 避免重复缩放
    )
    # 将填充的 token 设置为 -100 以忽略(这个应该设置成0)
    input_ids = inputs.input_ids
    input_ids[input_ids == processor.tokenizer.pad_token_id] = 0
    return inputs.pixel_values.to("cuda"), input_ids.to("cuda"), inputs.attention_mask.to("cuda")

# 随机选择验证集的子集（例如 20%）
val_subset_ratio = 0.2  # 使用 20% 的验证集
val_subset_size = int(val_subset_ratio * val_size)
val_subset_indices = random.sample(range(val_size), val_subset_size)
val_subset = torch.utils.data.Subset(val_dataset, val_subset_indices)

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_subset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# 使用标准的 CrossEntropyLoss
criterion = nn.CrossEntropyLoss(ignore_index=-100)

# 设置优化器
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)  # 权重衰减，防止过拟合

# 设置学习率调度器
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

# 记录损失和性能指标
train_losses = []
val_losses = []
bleu_scores = []  # 记录 BLEU 分数
spice_scores = []  # 记录 SPICE 分数

# 早停设置
best_val_loss = float("inf")
patience = 3  # 允许验证损失不下降的 epoch 数
early_stop_counter = 0

def calculate_bleu_score(model, dataloader, processor):
    model.eval()
    smoothie = SmoothingFunction().method4  # 使用平滑函数
    total_bleu = 0.0
    num_samples = 0
    
    for pixel_values, input_ids, attention_mask in dataloader:
        if pixel_values is None or input_ids is None:
            continue
        
        with torch.no_grad():
            # 生成描述
            generated_ids = model.generate(pixel_values=pixel_values)
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # 获取真实描述
            # 将 -100 转换回 pad_token_id 以便解码
            true_input_ids = input_ids.clone()
            true_input_ids[true_input_ids == -100] = processor.tokenizer.pad_token_id
            true_texts = processor.batch_decode(true_input_ids, skip_special_tokens=True)
            
            # 计算 BLEU 分数
            for generated, true in zip(generated_texts, true_texts):
                # 跳过空描述
                if not generated.strip() or not true.strip():
                    continue
                reference = [true.split()]  # 参考描述需要是列表的列表
                candidate = generated.split()  # 生成描述是一个词汇列表
                bleu = sentence_bleu(reference, candidate, smoothing_function=smoothie)
                total_bleu += bleu
                num_samples += 1
    
    return total_bleu / num_samples if num_samples > 0 else 0.0

# 初始化 SPICE 计算器
spice_scorer = Spice()

def calculate_spice_score(model, dataloader, processor):
    model.eval()
    total_spice = 0.0
    num_samples = 0
    
    # 使用 tqdm 显示进度条
    progress_bar = tqdm(dataloader, desc="SPICE 评估", leave=False)
    
    for pixel_values, input_ids, attention_mask in progress_bar:
        if pixel_values is None or input_ids is None:
            continue
        
        with torch.no_grad():
            # 生成描述
            generated_ids = model.generate(pixel_values=pixel_values)
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # 获取真实描述
            # 将 -100 转换回 pad_token_id 以便解码
            true_input_ids = input_ids.clone()
            true_input_ids[true_input_ids == -100] = processor.tokenizer.pad_token_id
            true_texts = processor.batch_decode(true_input_ids, skip_special_tokens=True)
            
            # 计算 SPICE 分数
            for generated, true in zip(generated_texts, true_texts):
                # 跳过空描述
                if not generated.strip() or not true.strip():
                    continue
                reference = [true]  # 参考描述
                candidate = generated  # 生成描述
                # 调用 compute_score 方法
                spice_score, _ = spice_scorer.compute_score({0: reference}, {0: [candidate]})
                total_spice += spice_score
                num_samples += 1
    
    return total_spice / num_samples if num_samples > 0 else 0.0

# 训练循环
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0

    # 随机采样 20% 的训练数据
    subset_size = int(0.2 * len(train_dataset))
    subset_indices = random.sample(range(len(train_dataset)), subset_size)
    train_subset = torch.utils.data.Subset(train_dataset, subset_indices)
    train_dataloader_subset = DataLoader(train_subset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    progress_bar = tqdm(train_dataloader_subset, desc=f"Epoch {epoch + 1}/{num_epochs} [训练]", leave=False)
    for pixel_values, input_ids, attention_mask in progress_bar:
        if pixel_values is None or input_ids is None or attention_mask is None:
            continue

        # 前向传播
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        loss = outputs.loss

        # 检查 loss 是否为 None
        if loss is None:
            print("警告：损失为 None，跳过当前 batch")
            continue

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 记录损失
        epoch_train_loss += loss.item()
        progress_bar.set_postfix({"训练损失": loss.item()})
    
    # 计算平均训练损失
    avg_train_loss = epoch_train_loss / len(train_dataloader_subset)
    train_losses.append(avg_train_loss)
    
    # 验证循环
    model.eval()
    epoch_val_loss = 0
    val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [验证]", leave=False)
    for pixel_values, input_ids, attention_mask in val_progress_bar:
        if pixel_values is None or input_ids is None or attention_mask is None:
            continue
        
        with torch.no_grad():
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            loss = outputs.loss
            epoch_val_loss += loss.item()
            val_progress_bar.set_postfix({"验证损失": loss.item()})
    
    # 计算平均验证损失
    avg_val_loss = epoch_val_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)
    
    # 更新学习率
    scheduler.step(avg_val_loss)
    
    # 计算 BLEU 分数
    bleu_score = calculate_bleu_score(model, val_dataloader, processor)
    bleu_scores.append(bleu_score)
    
    # 每隔 5 个 epoch 计算一次 SPICE 分数
    if (epoch + 1) % 5 == 0:
        spice_score = calculate_spice_score(model, val_dataloader, processor)
        spice_scores.append(spice_score)
        print(f"Epoch [{epoch + 1}/{num_epochs}], SPICE 分数: {spice_score:.4f}")
    else:
        # 如果不是计算 SPICE 的 epoch，则填充一个占位值（例如 None 或 0）
        spice_scores.append(None)
    
    # 打印性能指标
    print(f"Epoch [{epoch + 1}/{num_epochs}], 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}, BLEU 分数: {bleu_score:.4f}")
    
    # 每 5 个 epoch 保存一次模型
    if (epoch + 1) % 5 == 0:
        checkpoint_dir = os.path.join(current_dir, f"checkpoint_epoch_{epoch + 1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)
        print(f"模型已保存到: {checkpoint_dir}")
    
    # 早停逻辑
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        # 保存最佳模型
        model.save_pretrained(os.path.join(current_dir, "new_best_model"))
        processor.save_pretrained(os.path.join(current_dir, "new_best_model"))
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"验证损失在 {patience} 个 epoch 内未下降，提前停止训练。")
            break

# 绘制损失和 BLEU 分数曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label="train_losses")
plt.plot(range(1, len(val_losses) + 1), val_losses, label="validation_losses")
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.title("train_and_validation_loss_curves")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(bleu_scores) + 1), bleu_scores, label="BLEU_scores", color="green")
plt.xlabel("Epoch")
plt.ylabel("BLEU_scores")
plt.title("BLEU_Score_Change_Curve")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("training_metrics.png")  # 保存图表
plt.show()

# 绘制 SPICE 分数变化曲线
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(spice_scores) + 1), spice_scores, label="SPICE_scores", color="red")
plt.xlabel("Epoch")
plt.ylabel("SPICE_scores")
plt.title("SPICE_Score_Change_Curve")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("spice_score_curve.png")  # 保存图表
plt.show()

# 测试模型
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs)
        # out = model.generate(**inputs, max_length=50)  # 设置 max_length
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# 测试一张图片
test_image_path = os.path.join(data_dir, "images", "test_image.jpg")  # 替换为测试图片路径
caption = generate_caption(test_image_path)
print(f"生成的描述: {caption}")
