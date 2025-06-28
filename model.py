import os
import zipfile
import shutil
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Extract dataset
zip_path = "archive.zip"
extract_path = "plant_dataset"

if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# Step 2: Create subset of 2000 images
subset_path = "plant_dataset_subset"
max_images = 2000

if not os.path.exists(subset_path):
    os.makedirs(subset_path, exist_ok=True)
    image_root = next(Path(extract_path).rglob("*/*/*.jpg")).parent.parent

    all_classes = [d for d in image_root.iterdir() if d.is_dir()]
    images_per_class = max_images // len(all_classes)

    for cls_dir in all_classes:
        images = list(cls_dir.glob("*.jpg"))[:images_per_class]
        dest = Path(subset_path) / cls_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for img in images:
            shutil.copy(img, dest)

# Step 3: Load data
img_size = 224
batch_size = 32
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    subset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    subset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Step 4: Build model
base_model = tf.keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 5: Train and save
model.fit(train_gen, validation_data=val_gen, epochs=10)
model.save("plant_disease_model.h5")
print("✅ Model saved.")
