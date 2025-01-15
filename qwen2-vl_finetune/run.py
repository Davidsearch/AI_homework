# 导入库
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from flask import Flask, request, jsonify
import io



app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # 检查请求中是否包含文件
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    # 获取上传的文件
    file = request.files['file']

    # 将文件读取为图片



    image_path = "c.jpg"  # 替换为你的测试图像路径
    # image = Image.open(image_path).convert("RGB")  # 确保图像是 RGB 格式
    image = Image.open(file.stream).convert("RGB")
    target_size = (280, 280)  # 你可以根据需要调整尺寸
    image = image.resize(target_size)

    messages = [
        {
            "role": "user",
            "content": [
            {"type": "image", "image": image_path},  # 图像路径
            {"type": "text", "text": "描述该图像"},  # 提示文本
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],  # 直接传入 PIL 图像
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)


    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    # 解码生成的文本
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print("Generated Text:", output_text[0])


    return jsonify({"result":output_text[0]})


if __name__ == '__main__':


# 直接加载模型和处理器
# model_path = "Qwen2-VL"  # 替换为你的模型路径
    model_path = "fine-tuned-lora-model"
    model = AutoModelForVision2Seq.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)

# 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    app.run(host='0.0.0.0', port=5000)