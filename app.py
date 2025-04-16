from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from frontend

# Load the model (assumes model is in the same directory)
MODEL_PATH = "cats_vs_dogs_model.h5"
model = load_model("cats_vs_dogs_model.h5")

@app.route('/')
def home():
    return '✅ Cat vs Dog Classifier Backend is Live!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        img = Image.open(file.stream).convert("RGB").resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 150, 150, 3)
        prediction = model.predict(img_array)

        result = "Dog 🐶" if prediction[0][0] > 0.5 else "Cat 🐱"
        confidence = round(float(prediction[0][0]), 2)
        return jsonify({"prediction": result, "confidence": confidence})

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": "Prediction failed", "details": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
