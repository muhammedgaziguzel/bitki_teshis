from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Modeli yükleyelim
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "src", "saved_model", "plant_disease_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ['Healthy', 'Powdery', 'Rust']

def preprocess_image(image):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        print(f"İşlenen Görsel Boyutu: {image.shape}, Tip: {type(image)}")
        image = image / 255.0
        return np.expand_dims(image, axis=0)
    except Exception as e:
        print(f"Görsel işleme hatası: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Görsel gönderilmedi'}), 400

    file = request.files['image']
    image_np = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'error': 'Geçersiz görsel formatı'}), 400

    print(f"Gelen Görsel Boyutu: {image.shape}")

    label, confidence = predict_image(image)

    return jsonify({'label': label, 'confidence': confidence})

def predict_image(image):
    processed = preprocess_image(image)

    if processed is None:
        return "Unknown", 0.0

    print(f"Model Tahmin İçin Veri: {processed.shape}")

    prediction = model.predict(processed)[0]
    index = np.argmax(prediction)
    confidence = float(prediction[index])
    return CLASS_NAMES[index], confidence

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
