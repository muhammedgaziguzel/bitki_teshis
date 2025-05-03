import os
from src.train import train
from src.predict import start_camera

if __name__ == "__main__":
    # Model yoksa eğit
    if not os.path.exists("plant_disease_model.h5"):
        train()
    
    # Kamera açıp tahmin yap
    start_camera()
