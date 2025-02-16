from flask import Flask, request, render_template
import os
import cv2
import numpy as np
import pickle
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Load the ML model (Replace 'model.pkl' with your trained model)
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Preprocess image for prediction
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Resize to match model input
    img = img.flatten()  # Flatten for ML model
    return img.reshape(1, -1)

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Preprocess and predict
            img_data = preprocess_image(filepath)
            prediction = model.predict(img_data)[0]

            # Remedies based on prediction
            remedies = {
                "Healthy": "Your plant is healthy. Keep up the good care!",
                "Early Blight": "Use fungicide spray and remove infected leaves.",
                "Late Blight": "Apply copper-based fungicides and avoid watering leaves."
            }

            return render_template("index.html", filename=filename, prediction=prediction, remedy=remedies.get(prediction, "No remedy found."))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
