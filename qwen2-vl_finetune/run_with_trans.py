import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, MarianMTModel, MarianTokenizer
from PIL import Image
from flask import Flask, request, jsonify
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load vision-to-text model and processor
vision_model_path = "fine-tuned-lora-model"  # 替换为你的视觉模型路径
vision_model = AutoModelForVision2Seq.from_pretrained(vision_model_path)
vision_processor = AutoProcessor.from_pretrained(vision_model_path)

# Load translation model and tokenizer
translation_model_name = "Helsinki-NLP/opus-mt-en-zh"  # 英翻中模型
translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translation_model = MarianMTModel.from_pretrained(translation_model_name)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vision_model.to(device)
translation_model.to(device)
vision_model.eval()
translation_model.eval()

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def translate_text(text):
    """将英文文本翻译成中文"""
    inputs = translation_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    inputs = inputs.to(device)
    with torch.no_grad():
        translated_ids = translation_model.generate(**inputs)
    translated_text = translation_tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    return translated_text

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logging.info("Received a prediction request")
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        # Open and process the image
        image = Image.open(file.stream).convert("RGB")
        target_size = (280, 280)
        image = image.resize(target_size)

        # Prepare input for vision model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image in English."},  # 提示生成英文描述
                ],
            }
        ]

        text = vision_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = vision_processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Generate English description
        with torch.no_grad():
            generated_ids = vision_model.generate(**inputs, max_new_tokens=128)

        # Decode generated English text
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        english_text = vision_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print(english_text)
        logging.info(f"Generated English Text: {english_text}")

        # Translate English to Chinese
        chinese_text = translate_text(english_text)
        logging.info(f"Translated Chinese Text: {chinese_text}")

        return jsonify({"english_text": english_text, "chinese_text": chinese_text})
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)