from flask import Flask, request, jsonify
from flask_cors import CORS  # 新增
import os
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, MarianMTModel, MarianTokenizer

# 创建 Flask 应用
app = Flask(__name__)
CORS(app)  # 启用跨域支持
# 设置路径
current_dir = os.getcwd()  # 当前目录
model_dir = os.path.join(current_dir, "best_model")  # 模型保存目录
test_image_path = os.path.join(current_dir, "uploads/photo.png")  # 测试图片路径

# 加载训练后的模型和分词器
processor = BlipProcessor.from_pretrained(model_dir)
model = BlipForConditionalGeneration.from_pretrained(model_dir).to("cuda")

# 加载英文到中文的翻译模型和分词器
translation_model_name = "./opus-mt-en-zh"
translator_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translator_model = MarianMTModel.from_pretrained(translation_model_name).to("cuda")

# 测试图片生成描述
def generate_caption(image_path):
    """
    生成图片的描述
    :param image_path: 图片路径
    :return: 生成的描述文本
    """
    # 加载图片
    image = Image.open(image_path).convert("RGB")
    
    # 使用处理器预处理图片
    inputs = processor(image, return_tensors="pt").to("cuda")
    
    # 使用模型生成描述
    with torch.no_grad():
        out = model.generate(**inputs, max_length=50)  # 设置 max_length
    
    # 解码生成的描述
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def translate_to_chinese(text):
    """
    将英文文本翻译成中文
    :param text: 英文文本
    :return: 中文文本
    """
    inputs = translator_tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to("cuda")
    with torch.no_grad():
        translated = translator_model.generate(**inputs)
    return translator_tokenizer.decode(translated[0], skip_special_tokens=True)

# 定义网页内容
html = ''' 
<html>
    <head><title>Simple Web Server</title></head>
    <body>
        <h1>404</h1>
        <p>can't find</p>
    </body>
</html>
'''

# 定义路由 /index，返回一个网页
@app.route('/')
def index():
    return html

@app.route('/predict', methods=['POST'])
def predict():
    # 检查是否有文件上传
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    # 检查文件是否为空
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # 保存文件到本地
    save_path = os.path.join('uploads', file.filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file.save(save_path)
    print(f'File saved to: {save_path}')
    
    # 生成图片描述
    caption = generate_caption(save_path)
    print(f'Generated Caption: {caption}')

    # 翻译成中文
    translated_caption = translate_to_chinese(caption)
    print(f'Translated Caption: {translated_caption}')

    # 返回结果
    return jsonify({'message': translated_caption}), 200

# 定义路由 /hello，调用函数返回 "Hello, World!"
@app.route('/hello')
def hello():
    return 'Hello, World!'

# 启动服务器
if __name__ == '__main__':
    with open('index_1.html', "r") as file:
        html = file.read()
    app.run(host='0.0.0.0', port=4321)
