import tensorflow as tf
import numpy as np
import cv2
import os
from tkinter import filedialog, Tk, Button, Label, Text, Scrollbar, Frame, BOTH, END
from PIL import Image, ImageTk
from flask import Flask, request, jsonify  # Flask eklendi

CLASS_NAMES = ['Healthy', 'Powdery', 'Rust']

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "src", "saved_model", "plant_disease_model.h5")

DISEASE_INFO = {
    'Healthy': {
        'description': '🌿 Bitkiniz sağlıklı!',
        'solution': 'Mevcut bakım rutininize devam edin.'
    },
    'Powdery': {
        'description': '⚠️ Külleme hastalığı tespit edildi.',
        'solution': '• Etkilenen yaprakları budayın\n• Havalandırmayı artırın\n• Organik ilaçlar uygulayın'
    },
    'Rust': {
        'description': '⚠️ Pas hastalığı tespit edildi.',
        'solution': '• Hasta yaprakları kesin\n• Fungisit kullanın\n• Hava sirkülasyonunu artırın'
    }
}

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def predict_image(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0]
    index = np.argmax(prediction)
    confidence = float(prediction[index])
    return CLASS_NAMES[index], confidence

def display_result(image, label, confidence):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb).resize((400, 400))
    photo = ImageTk.PhotoImage(image_pil)

    image_label.config(image=photo)
    image_label.image = photo

    result_label.config(text=f"{label.upper()}")

    info = DISEASE_INFO.get(label, {})
    solution_textbox.delete(1.0, END)
    if label == 'Healthy':
        solution_textbox.insert(END, info['description'])
    else:
        solution_textbox.insert(END, f"{info['description']}\n\n{info['solution']}")

def select_image():
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not filepath:
        return
    image = cv2.imread(filepath)
    if image is None:
        result_label.config(text="❌ Görsel yüklenemedi.")
        return
    label, confidence = predict_image(image)
    display_result(image, label, confidence)

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        result_label.config(text="❌ Kamera açılamadı.")
        return
    ret, frame = cap.read()
    cap.release()
    if not ret:
        result_label.config(text="❌ Görüntü alınamadı.")
        return
    label, confidence = predict_image(frame)
    display_result(frame, label, confidence)

# Flask API bölümü eklendi
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_api():
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "Görsel bulunamadı"}), 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"status": "error", "message": "Görsel çözümlenemedi"}), 400

    label, confidence = predict_image(img)

    return jsonify({
        "status": "success",
        "label": label,
        "confidence": round(confidence, 4),
        "description": DISEASE_INFO[label]["description"],
        "solution": DISEASE_INFO[label]["solution"]
    })

# Tkinter GUI kısmı
app_tk = Tk()
app_tk.title("🌱 Bitki Hastalığı Tanı Sistemi")
app_tk.geometry("600x800")
app_tk.configure(bg="#f8f9fa")

def style_button(btn, color):
    btn.configure(
        bg=color,
        fg="white",
        font=("Segoe UI", 11, "bold"),
        activebackground=color,
        relief="flat",
        padx=20,
        pady=8,
        bd=0,
        cursor="hand2"
    )

Label(
    app_tk,
    text="Yaprak görseli seçin ya da fotoğraf çekin",
    font=("Segoe UI", 14, "bold"),
    bg="#f8f9fa",
    fg="#212529"
).pack(pady=25)

btn_select = Button(app_tk, text="📁 Görsel Seç", command=select_image)
style_button(btn_select, "#28a745")
btn_select.pack(pady=5)

btn_camera = Button(app_tk, text="📷 Kameradan Çek", command=capture_image)
style_button(btn_camera, "#007bff")
btn_camera.pack(pady=5)

image_label = Label(app_tk, bg="#dee2e6", width=400, height=400, relief="groove", bd=2)
image_label.pack(pady=25)

result_label = Label(app_tk, text="", font=("Segoe UI", 16, "bold"), bg="#f8f9fa", fg="#343a40")
result_label.pack(pady=10)

Label(app_tk, text="🩺 Tanı ve Çözüm", font=("Segoe UI", 14, "bold"), bg="#f8f9fa", fg="#212529").pack()

text_frame = Frame(app_tk, bg="#f8f9fa")
text_frame.pack(padx=20, pady=10, fill=BOTH, expand=True)

scroll = Scrollbar(text_frame)
scroll.pack(side="right", fill="y")

solution_textbox = Text(
    text_frame,
    height=8,
    wrap="word",
    font=("Segoe UI", 11),
    yscrollcommand=scroll.set,
    bg="#ffffff",
    fg="#212529",
    relief="solid",
    bd=1
)
solution_textbox.pack(fill=BOTH, expand=True)
scroll.config(command=solution_textbox.yview)

# Flask'i ayrı thread’de başlatmak gerekebilir (opsiyonel)
# Eğer GUI ve API aynı anda çalışsın istersen bunu uygula
# from threading import Thread
# Thread(target=app.run, kwargs={"port": 5000}).start()

app_tk.mainloop()
