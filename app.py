import numpy as np
import tensorflow as tf
from tensorflow import keras
import gradio as gr

path = '../input/plantvillage-dataset/color'
train_ds = keras.utils.image_dataset_from_directory(
    path,
    image_size=(224, 224),
    batch_size=32,
    seed=123,
    validation_split=0.2,
    subset='training'
)

test_ds = keras.utils.image_dataset_from_directory(
    path,
    image_size=(224, 224),
    batch_size=32,
    seed=123,
    validation_split=0.2,
    subset='validation'
)

classes = train_ds.class_names

model = keras.Sequential([
    keras.layers.Rescaling(1/255, input_shape=(224, 224, 3)),
    keras.layers.Conv2D(32, (3,3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(classes), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10
)


def predict_image(image):
    img = tf.image.resize(image, (224, 224))
    img = tf.expand_dims(img, 0) / 255.0
    preds = model.predict(img)
    idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    class_name = classes[idx]
    if "___" in class_name:
        plant, disease = class_name.split("___", 1)
    else:
        plant, disease = class_name, "Healthy"
    return plant, disease if disease.lower() != "healthy" else "Healthy", f"{confidence*100:.2f}%"

demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Textbox(label="Plant Name"),
        gr.Textbox(label="Disease"),
        gr.Textbox(label="Confidence")
    ],
    title="Plant Disease Detection",
    description="Upload a plant leaf image to identify the plant and detect if it's healthy or diseased."
)
demo.launch()

