from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
model = tf.keras.models.load_model("plant_disease_model.h5")

# You can modify this if you know your class names
class_names = list(os.listdir("plant_dataset_subset"))
class_names.sort()

def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        file = request.files["image"]
        if file:
            image = Image.open(file.stream)
            processed = preprocess_image(image)
            pred = model.predict(processed)
            predicted_class = class_names[np.argmax(pred)]
            confidence = round(100 * np.max(pred), 2)
            prediction = f"{predicted_class} ({confidence}%)"
    return render_template("index.html", prediction=prediction)
    
if __name__ == "__main__":
    app.run(debug=True)
