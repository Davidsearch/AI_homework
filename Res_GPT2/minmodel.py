import torch
from torchvision import models, transforms
from PIL import Image
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# 1. 加载保存的模型
class ResNetGPT2(torch.nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(ResNetGPT2, self).__init__()
        # 定义模型结构（与训练时一致）
        self.cnn = torch.nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])
        self.image_embedding = torch.nn.Linear(2048, embed_size)
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        self.image_to_gpt2 = torch.nn.Linear(embed_size, self.gpt2.config.hidden_size)

    def forward(self, images, input_ids=None, attention_mask=None):
        features = self.cnn(images)
        features = features.view(features.size(0), -1)
        features = self.image_embedding(features)
        image_embeddings = self.image_to_gpt2(features)
        if input_ids is None:
            outputs = self.gpt2(inputs_embeds=image_embeddings.unsqueeze(1), attention_mask=attention_mask)
        else:
            seq_length = input_ids.size(1)
            inputs_embeds = image_embeddings.unsqueeze(1).repeat(1, seq_length, 1)
            outputs = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return outputs.logits

# 2. 加载保存的模型权重
model_path = "./model_epoch_120.pth"  # 替换为你的模型路径
embed_size = 256
hidden_size = 512
model = ResNetGPT2(embed_size, hidden_size)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # 加载模型权重
model.eval()  # 设置为评估模式

# 3. 加载 GPT-2 的 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# 4. 图像预处理
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 增加 batch 维度
    return image

# 5. 生成图像描述
def generate_caption(image_path, max_length=20):
    # 预处理图像
    image = preprocess_image(image_path)
    
    # 提取图像特征
    with torch.no_grad():
        features = model.cnn(image)
        features = features.view(features.size(0), -1)
        features = model.image_embedding(features)
        image_embeddings = model.image_to_gpt2(features)
    
    # 初始化输入（GPT-2 的起始 token）
    input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(image.device)
    
    # 逐步生成描述
    caption = []
    for _ in range(max_length):
        with torch.no_grad():
            # 仅使用 inputs_embeds 作为输入
            outputs = model.gpt2(inputs_embeds=image_embeddings.unsqueeze(1))
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            
            # 如果生成结束 token，停止生成
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            
            # 将 token 转换为单词
            word = tokenizer.decode(next_token_id.item(), skip_special_tokens=True)
            caption.append(word)
    
    return ' '.join(caption)

# 6. 测试模型
image_path = "a.jpg"  # 替换为你的测试图像路径
caption = generate_caption(image_path, max_length=20)
print(f"Generated Caption: {caption}")