import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image
import json
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
tokenizer_GPT = GPT2Tokenizer.from_pretrained('gpt2')

# class ResNetGPT2(nn.Module):
#     def __init__(self, embed_size, hidden_size):
#         super(ResNetGPT2, self).__init__()
        
#         # 使用预训练的ResNet提取图像特征
#         self.cnn = models.resnet50(pretrained=True)
#         self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])  # 去掉全连接层
        
#         # 图像特征到嵌入向量的映射
#         self.image_embedding = nn.Linear(2048, embed_size)
        
#         # 使用GPT-2作为语言模型
#         self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
#         self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
#         # 将图像特征映射到GPT-2的输入空间
#         self.image_to_gpt2 = nn.Linear(embed_size, self.gpt2.config.hidden_size)
        
#     def forward(self, images, captions):
#         # 提取图像特征
#         features = self.cnn(images)
#         features = features.view(features.size(0), -1)  # 展平特征
#         features = self.image_embedding(features)
        
#         # 将图像特征映射到GPT-2的输入空间
#         image_embeddings = self.image_to_gpt2(features)
        
#         # 将图像特征和caption输入GPT-2
#         outputs = self.gpt2(inputs_embeds=image_embeddings, labels=captions)
#         return outputs.logits

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

def preprocess_image(image_path):
    # 图像预处理：调整大小、归一化等
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 增加batch维度
    return image

def generate_caption(model, image_path, max_length=20):
    model.eval()
    image = preprocess_image(image_path)
    
    # 提取图像特征
    with torch.no_grad():
        features = model.cnn(image)
        features = features.view(features.size(0), -1)
        features = model.image_embedding(features)
        image_embeddings = model.image_to_gpt2(features)
    
    # 初始化输入（GPT-2的起始token）
    input_ids = torch.tensor([[model.gpt2_tokenizer.bos_token_id]]).to(image.device)
    
    # 逐步生成描述
    caption = []
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model.gpt2(input_ids=input_ids, inputs_embeds=image_embeddings)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            
            # 如果生成结束token，停止生成
            if next_token_id.item() == model.gpt2_tokenizer.eos_token_id:
                break
            
            # 将token转换为单词
            word = model.gpt2_tokenizer.decode(next_token_id.item(), skip_special_tokens=True)
            caption.append(word)
    
    return ' '.join(caption)


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
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # 将 eos_token 设置为 pad_token
def collate_fn(batch):
    # 分离图像和描述
    images, captions = zip(*batch)
    
    # 将图像堆叠成一个张量
    images = torch.stack(images, dim=0)
    
    # 使用 tokenizer 对描述进行编码和填充
    inputs = tokenizer(captions, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    return images, input_ids, attention_mask


# 加载单张图片
def load_and_preprocess_image(image_path, transform):
    """
    加载并预处理单张图片。
    Args:
        image_path (str): 图片路径。
        transform (callable): 图片预处理函数。
    Returns:
        image (torch.Tensor): 预处理后的图片，形状为 (1, 3, 224, 224)。
    """
    # 加载图片
    image = Image.open(image_path).convert('RGB')
    
    # 预处理图片
    if transform:
        image = transform(image)
    # print(image.shape)
    image = image.unsqueeze(0)
    return image

# 测试单张图片
def test_single_image(model, image_path, transform, device):
    """
    测试单张图片并生成描述。
    Args:
        model (nn.Module): 训练好的模型。
        image_path (str): 图片路径。
        transform (callable): 图片预处理函数。
        device (torch.device): 设备（如 'cuda' 或 'cpu'）。
    Returns:
        caption (str): 生成的描述。
    """
    # 加载并预处理图片
    image = load_and_preprocess_image(image_path, transform)
    image = image.to(device)  # 将图片移动到设备
    
    # 将模型设置为评估模式
    model.eval()
    model.to(device)

    # 生成描述
    with torch.no_grad():
        outputs = model(image)  # 调用模型的 forward 方法
        logits = outputs  # 获取 logits
        
        # 将 logits 转换为 token IDs
        predicted_token_ids = torch.argmax(logits, dim=-1)
        
        # 将 token IDs 转换为文本
        caption = tokenizer_GPT.decode(predicted_token_ids[0], skip_special_tokens=True)
    
    return caption


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
dataloader = DataLoader(dataset, batch_size=8, shuffle=True,collate_fn=collate_fn)


import torch.optim as optim
from tqdm import tqdm  # 导入 tqdm

# 初始化模型
embed_size = 256
hidden_size = 512
model = ResNetGPT2(embed_size, hidden_size)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
# 训练循环
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0  # 用于记录每个 epoch 的总损失
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)  # 创建进度条
    
    for images, input_ids, attention_mask in progress_bar:
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # 前向传播
        outputs = model(images, input_ids=input_ids, attention_mask=attention_mask)
        
        # 计算损失
        logits = outputs.view(-1, outputs.size(-1))  # (batch_size * sequence_length, vocab_size)
        labels = input_ids.view(-1)  # (batch_size * sequence_length)
        loss = criterion(logits, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 更新进度条显示
        epoch_loss += loss.item()
        progress_bar.set_postfix({"Loss": loss.item()})  # 显示当前 batch 的损失
    
    # 输出每个 epoch 的平均损失
    avg_loss = epoch_loss / len(dataloader)
    image_path = "data/test_image.jpg"  # 替换为你的测试图片路径
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
    caption = test_single_image(model, image_path, transform, device)
    print(f"Generated Caption: {caption}")
    model.save_pretrained("./student-model_"+str(epoch))

