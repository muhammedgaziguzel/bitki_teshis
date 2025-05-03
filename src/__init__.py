# Bu dosya BOŞ kalacak (sadece projenizin Python paketi olarak tanınması için)
import tensorflow as tf

class GPUPredictor:
    def __init__(self, model_path='plant_disease_model.h5'):  # veya 'best_model.h5'
        # GPU ayarları
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = ['Healthy', 'Powdery', 'Rust']
