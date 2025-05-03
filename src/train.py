import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Dataset klasörünü doğru tanımla
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Model kayıt dizini
SAVED_MODEL_DIR = os.path.join(BASE_DIR, "src", "saved_model")
os.makedirs(SAVED_MODEL_DIR, exist_ok=True)

# Parametreler
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20  # Epoch sayısı artırıldı
NUM_CLASSES = 3

def load_datasets():
    train_dir = os.path.join(DATASET_DIR, "Train")
    val_dir = os.path.join(DATASET_DIR, "Validation")

    # Data augmentation (EKLENDİ)
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_ds = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_ds = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_ds, val_ds

def build_model():
    # Transfer Learning (EKLENDİ)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze the base model

    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Regularization (EKLENDİ)
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

def train():
    print("Veriler yükleniyor...")
    train_ds, val_ds = load_datasets()

    print(f"Sınıflar: {train_ds.class_indices}")

    print("Model oluşturuluyor...")
    model = build_model()
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Learning rate ayarlandı
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks (GELİŞTİRİLDİ)
    checkpoint = ModelCheckpoint(
        os.path.join(SAVED_MODEL_DIR, "plant_disease_model.h5"),
        save_best_only=True,
        monitor='val_accuracy'
    )
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

    print("Model eğitiliyor...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping]
    )

    # Model kaydediliyor
    model.save(os.path.join(SAVED_MODEL_DIR, "plant_disease_model.h5"))
    print(f"Model kaydedildi: {SAVED_MODEL_DIR}/plant_disease_model.h5")

if __name__ == "__main__":
    train()