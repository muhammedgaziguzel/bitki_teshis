import tensorflow as tf
import os
import matplotlib.pyplot as plt

def load_datasets(batch_size=32, image_size=(224, 224)):
    # Klasör yolları
    train_dir = 'dataset/Train'
    val_dir = 'dataset/Validation'
    test_dir = 'dataset/Test'

    # Veri artırma işlemleri
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomHeight(0.2),
        tf.keras.layers.RandomWidth(0.2)
    ])

    # Eğitim verisini yükle
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True
    )

    # Doğrulama verisini yükle
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True
    )

    # Test verisini yükle
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False
    )

    # Eğitim setine veri artırma uygula
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE)

    # Performans artırıcı önbellekleme ve ön getirme
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds

def plot_class_distribution(train_dir):
    class_counts = {}
    for label in os.listdir(train_dir):
        class_path = os.path.join(train_dir, label)
        if os.path.isdir(class_path):
            class_counts[label] = len(os.listdir(class_path))

    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel("Sınıflar")
    plt.ylabel("Görüntü Sayısı")
    plt.title("Eğitim Seti Sınıf Dağılımı")
    plt.xticks(rotation=45)
    plt.show()
