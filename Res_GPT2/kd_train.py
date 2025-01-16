import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoProcessor, AutoModelForSeq2SeqLM, AdamW, get_scheduler, AutoModelForVision2Seq, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from PIL import Image
import os
from qwen_vl_utils import process_vision_info
from transformers import BertModel

from torchvision import models, transforms
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image

import json

class ResNetGPT2(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(ResNetGPT2, self).__init__()
        
        # 使用预训练的ResNet提取图像特征
        self.cnn = models.resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])  # 去掉全连接层
        
        # 图像特征到嵌入向量的映射
        self.image_embedding = nn.Linear(2048, embed_size)
        
        # 使用GPT-2作为语言模型
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        
        # 将图像特征映射到GPT-2的输入空间
        self.image_to_gpt2 = nn.Linear(embed_size, self.gpt2.config.hidden_size)
        
  
        '''
    Args:
        images (torch.Tensor): 输入图像，形状为 (batch_size, 3, 224, 224)
        input_ids (torch.Tensor, optional): 输入的 token IDs，形状为 (batch_size, sequence_length)
        attention_mask (torch.Tensor, optional): 注意力掩码，形状为 (batch_size, sequence_length)
    Returns:
        logits (torch.Tensor): 模型输出的 logits，形状为 (batch_size, sequence_length, vocab_size)
    '''
    def forward(self, images, input_ids=None, attention_mask=None):
        # 提取图像特征
        features = self.cnn(images)
        features = features.view(features.size(0), -1)  # 展平特征
        features = self.image_embedding(features)
    
        # 将图像特征映射到GPT-2的输入空间
        image_embeddings = self.image_to_gpt2(features)
    
        # 如果 input_ids 为 None，说明是推理模式（仅输入图像）
        if input_ids is None:
            # 推理模式：仅使用图像特征作为输入
            outputs = self.gpt2(
                inputs_embeds=image_embeddings.unsqueeze(1),  # 将图像特征扩展为序列
                attention_mask=attention_mask  # 传入 attention_mask（可以为 None）
            )
        else:
            # 训练模式：将图像特征和文本输入 GPT-2
            # 扩展 inputs_embeds 的序列长度
            seq_length = input_ids.size(1)  # 获取 input_ids 的序列长度
            inputs_embeds = image_embeddings.unsqueeze(1).repeat(1, seq_length, 1)  # 复制图像嵌入向量
        
            outputs = self.gpt2(
                inputs_embeds=inputs_embeds,  # 扩展后的 inputs_embeds
                attention_mask=attention_mask  # 传入 attention_mask
            )
    
        return outputs.logits

    def save_pretrained(self, save_directory):
        """
        将模型保存到指定目录。
        Args:
            save_directory (str): 保存模型的目录路径。
        """
        # 如果目录不存在，则创建
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        # 保存 ResNetGPT2 的权重
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # 保存 GPT-2 的配置
        self.gpt2.save_pretrained(save_directory+"/gpt2")
        
        # 保存其他配置（如果需要）
        config = {
            "embed_size": self.image_embedding.out_features,
            "hidden_size": self.gpt2.config.hidden_size
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f)
        
        print(f"Model saved to {save_directory}")



    @classmethod
    def from_pretrained(cls, save_directory):
        """
        从指定目录加载模型。
        Args:
            save_directory (str): 保存模型的目录路径。
        Returns:
            model (ResNetGPT2): 加载的模型实例。
        """
        # 加载配置
        with open(os.path.join(save_directory, "config.json"), "r") as f:
            config = json.load(f)
    
        # 初始化模型
        model = cls(embed_size=config["embed_size"], hidden_size=config["hidden_size"])
    
        # 加载权重
        model.load_state_dict(torch.load(os.path.join(save_directory, "pytorch_model.bin")))
    
        # 加载 GPT-2 的配置
        model.gpt2.config = GPT2LMHeadModel.from_pretrained(save_directory).config
    
        print(f"Model loaded from {save_directory}")
        return model

# class SimpleVisionLanguageModel(nn.Module):
#     def __init__(self, visual_feature_dim=1176, hidden_dim=512, num_layers=4, vocab_size=151643):
#         super().__init__()
#         # 文本编码器（轻量级）
#         self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        
#         # 视觉编码器（直接处理特征向量）
#         self.visual_encoder = nn.Sequential(
#             nn.Linear(visual_feature_dim, hidden_dim),  # 将特征维度映射到隐藏维度
#             nn.ReLU(),
#         )
        
#         # 多模态融合模块
#         self.fusion_layer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8),
#             num_layers=num_layers,
#         )
        
#         # 输出层（生成文本）
#         self.output_layer = nn.Linear(hidden_dim, vocab_size)

#     def forward(self, images, input_ids, attention_mask):
#         # 提取文本特征

#         print("Input IDs shape:", input_ids.shape)  
#         print("Attention mask shape:", attention_mask.shape) 
#         text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
#         print("Text features shape:", text_features.shape)  
        
#         # 提取视觉特征
#         visual_features = self.visual_encoder(images)  # 输入形状: (batch_size, num_patches, feature_dim)
#         visual_features = visual_features.unsqueeze(1)  # 增加序列维度
        
#         # 拼接文本和视觉特征
#         combined_features = torch.cat([text_features, visual_features], dim=1)
        
#         # 多模态融合
#         fused_features = self.fusion_layer(combined_features)
        
#         # 生成输出
#         logits = self.output_layer(fused_features[:, 0, :])  # 使用 [CLS] 特征
#         return logits


# 加载教师模型（微调后的 Qwen2-VL）
teacher_model = AutoModelForVision2Seq.from_pretrained("./fine-tuned-lora-model", torch_dtype="auto", device_map="auto")
teacher_processor = AutoProcessor.from_pretrained("./fine-tuned-lora-model")
teacher_model.to("cuda")
teacher_model.eval()
tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-lora-model", use_fast=False, trust_remote_code=True)


# 加载小模型
embed_size = 256
hidden_size = 512

student_model = ResNetGPT2(embed_size, hidden_size)
student_model.to("cuda")


# 加载数据集（与微调代码中的数据集相同）
def load_data(data_dir):
    captions_path = os.path.join(data_dir, "captions.txt")
    with open(captions_path, "r") as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        image_path, text = line.strip().split(",", 1)
        data.append({"image_path": os.path.join(data_dir, image_path), "text": text})
    return data

# 数据预处理函数（与微调代码中的预处理函数相同）
def preprocess_function(examples):
    MAX_LENGTH = 8192
    output_content = examples["text"]
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": examples["image_path"], "resized_height": 280, "resized_width": 280},
                {"type": "text", "text": "描述该图片"},
            ],
        }
    ]
    
    text = teacher_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = teacher_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    inputs = {key: value.tolist() for key, value in inputs.items()}
    instruction = inputs

    response = tokenizer(examples["text"], add_special_tokens=False)

    
    input_ids = instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"][0]) + response["input_ids"] + [tokenizer.pad_token_id]

    
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)

    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": inputs['pixel_values'],
        "image_grid_thw": inputs['image_grid_thw']
    }

# 创建数据集类（与微调代码中的数据集类相同）
class ImageTextDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = preprocess_function(item)
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        return inputs

# 加载数据
data_dir = "data"  # 替换为你的数据文件夹路径
data = load_data(data_dir)

# 创建数据集实例
train_dataset = ImageTextDataset(data, teacher_processor)

# 创建 DataLoader
teacher_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False, collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True))












import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image




class ImageCaptionDataset(Dataset):
    def __init__(self, caption_file, image_dir, transform=None):
        """
        Args:
            caption_file (str): Path to the caption file.
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # 读取描述文件
        with open(caption_file, 'r') as f:
            lines = f.readlines()
        
        # 解析图像路径和描述
        self.image_captions = []
        for line in lines:
            parts = line.strip().split(',',maxsplit=1)  # 分割图像路径和描述
            if len(parts) == 2:
                image_path, caption = parts
                self.image_captions.append((image_path, caption))
    
    def __len__(self):
        return len(self.image_captions)
    
    def __getitem__(self, idx):
        image_path, caption = self.image_captions[idx]
        
        # 加载图像
        image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, caption

from transformers import GPT2Tokenizer

# 初始化 GPT-2 的 tokenizer
tokenizer_GPT = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer_GPT.pad_token = tokenizer_GPT.eos_token  # 将 eos_token 设置为 pad_token
def collate_fn(batch):
    # 分离图像和描述
    images, captions = zip(*batch)
    
    # 将图像堆叠成一个张量
    images = torch.stack(images, dim=0)
    
    # 使用 tokenizer 对描述进行编码和填充
    inputs = tokenizer_GPT(captions, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    return images, input_ids, attention_mask

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载数据集
caption_file = "data/captions.txt"
image_dir = "data"
dataset = ImageCaptionDataset(caption_file, image_dir, transform=transform)

# 创建数据加载器
student_dataloader = DataLoader(dataset, batch_size=8, shuffle=False,collate_fn=collate_fn)

student_vocab = tokenizer_GPT.get_vocab()
teacher_vocab = tokenizer.get_vocab()
vocab_map = {}
for token in student_vocab:
    if token in teacher_vocab:
        vocab_map[teacher_vocab[token]] = student_vocab[token]
        
student_vocab_size= len(student_vocab)

def align_logits(teacher_logits):
    """
    将教师模型的 logits 对齐到学生模型的词汇表空间
    :param teacher_logits: 教师模型的输出 logits，形状为 [batch_size, seq_len, teacher_vocab_size]
    :param vocab_map: 词汇表映射，键是教师模型的索引，值是学生模型的索引
    :param student_vocab_size: 学生模型的词汇表大小
    :return: 对齐后的 logits，形状为 [batch_size, seq_len, student_vocab_size]
    """
    batch_size, seq_len, _ = teacher_logits.shape
    
    # 初始化对齐后的 logits
    aligned_logits = torch.full(
        (batch_size, seq_len, student_vocab_size),
        fill_value=-float("inf"),
        device=teacher_logits.device
    )
    
    # 将教师模型的 logits 映射到学生模型的词汇表空间
    for teacher_idx, student_idx in vocab_map.items():
        if teacher_idx < teacher_logits.size(2) and student_idx < student_vocab_size:
            aligned_logits[:, :, student_idx] = teacher_logits[:, :, teacher_idx]
    
    return aligned_logits

# def align_logits(teacher_logits):
#     """
#     将教师模型的 logits 对齐到学生模型的词汇表空间
#     :param teacher_logits: 教师模型的输出 logits
#     :param teacher_vocab: 教师模型的词汇表
#     :param student_vocab: 学生模型的词汇表
#     :return: 对齐后的 logits
#     """
#     # 创建一个映射矩阵，将教师模型的词汇表索引映射到学生模型的词汇表索引

#     aligned_logits = torch.full(
#         (teacher_logits.size(0), len(student_vocab)),
#         fill_value=-float("inf"),
#         device=teacher_logits.device
#     )
    

#     for teacher_idx, student_idx in vocab_map.items():
#         aligned_logits[:, student_idx] = teacher_logits[:, teacher_idx]

#     return aligned_logits

def distillation_loss(student_logits, aligned_teacher_logits, temperature=5.0):
    """
    计算蒸馏损失（KL 散度）
    :param student_logits: 学生模型的输出 logits
    :param aligned_teacher_logits: 对齐后的教师模型 logits
    :param temperature: 温度参数，用于平滑概率分布
    :return: 蒸馏损失
    """
    # 使用 softmax 和温度参数平滑概率分布
    student_probs = F.softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(aligned_teacher_logits / temperature, dim=-1)
    
    # 计算 KL 散度
    loss = nn.KLDivLoss(reduction="batchmean")(torch.log(student_probs), teacher_probs)
    return loss

def combined_loss(student_logits, aligned_teacher_logits, labels, temperature=5.0, alpha=0.5):
    """
    计算联合损失
    :param student_logits: 学生模型的输出 logits
    :param aligned_teacher_logits: 对齐后的教师模型 logits
    :param labels: 真实标签
    :param temperature: 温度参数
    :param alpha: 任务损失的权重
    :return: 联合损失
    """
    # 计算蒸馏损失
    distill_loss = distillation_loss(student_logits, aligned_teacher_logits, temperature)
    # 计算任务损失
    task_loss_value = task_loss(student_logits, labels)
    # 联合损失
    loss = alpha * task_loss_value + (1 - alpha) * distill_loss
    return loss






# 设置训练参数
optimizer = AdamW(student_model.parameters(), lr=1e-4)
num_epochs = 3
num_training_steps = num_epochs * len(teacher_dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# 训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
student_model.to(device)

student_model.train()

student_model.save_pretrained("./distilled-student-model_ori")

# 训练过程
num_epochs = 10
for epoch in range(num_epochs):
    # student_model.save_pretrained("./distilled-student-model_ori"+str(epoch))
    teacher_model.eval()  # 教师模型在评估模式下
    student_model.train()  # 学生模型在训练模式下
    # epoch_loss = 0  # 用于记录每个 epoch 的总损失
    # progress_bar = tqdm(student_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)  # 创建进度条


    for (teacher_batch, student_batch) in zip(teacher_dataloader, student_dataloader):
        # 将数据移动到设备（如GPU）
        # print(teacher_inputs)
        teacher_inputs = {k: v.to(device) for k, v in teacher_batch.items()}
       
        images, input_ids, attention_mask=student_batch
        images=images.to(device)
        input_ids=input_ids.to(device)
        attention_mask=attention_mask.to(device)
        # student_inputs = {k: v.to(device) for k, v in student_batch.items()}


        with torch.no_grad():
            teacher_outputs = teacher_model(**teacher_inputs)
            teacher_logits = teacher_outputs.logits
        # 学生模型的前向传播
        student_outputs = student_model(images, input_ids=input_ids, attention_mask=attention_mask)
     
        teacher_logits = teacher_logits[:, :student_outputs.shape[1], :]  # 截断到序列长度 20

        aligned_teacher_logits = align_logits(
            teacher_logits
        )
        
        # 计算蒸馏损失
        loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student_outputs / 2.0, dim=-1),
            torch.nn.functional.softmax(aligned_teacher_logits / 2.0, dim=-1),
            reduction="batchmean",
        )
        
    
        # 反向传播和优化
        optimizer.zero_grad()


        loss.backward()
        optimizer.step()
        print(loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    student_model.save_pretrained("./distilled-student-model_"+str(epoch))

# for epoch in range(num_epochs):
#     for batch in train_dataloader:
#         batch = {k: v.to("cuda") for k, v in batch.items()}
#         print()
        
#         # 教师模型生成 soft labels
#         with torch.no_grad():
#             teacher_outputs = teacher_model(**batch)
#             teacher_logits = teacher_outputs.logits
#         print(teacher_logits)
#         # 学生模型前向传播
#         student_inputs = {
#             "images": batch["pixel_values"],  # 图像输入
#             "input_ids": batch["input_ids"],  # 文本输入
#             "attention_mask": batch["attention_mask"],  # 注意力掩码
#         }

#         student_outputs = student_model(batch["pixel_values"],batch["input_ids"],batch["attention_mask"])
#         student_logits = student_outputs
        
#         # 计算蒸馏损失（使用 KL 散度）
#         loss = torch.nn.functional.kl_div(
#             torch.nn.functional.log_softmax(student_logits / 2.0, dim=-1),
#             torch.nn.functional.softmax(teacher_logits / 2.0, dim=-1),
#             reduction="batchmean",
#         )
        
#         # 反向传播
#         loss.backward()
#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
        
#         # 打印损失
#         print(f"Epoch {epoch + 1}, Distillation Loss: {loss.item()}")

# 保存蒸馏后的学生模型

# student_processor.save_pretrained("./distilled-student-model") 