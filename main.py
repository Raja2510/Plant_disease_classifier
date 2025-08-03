from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 1️⃣ Load model at startup
model = load_model("ttrained_model.keras")
print("✅ Model loaded successfully!")

# Define class names
class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

app = FastAPI(title="Image Disease Prediction API")

@app.get("/")
def home():
    return {"message": "Keras Image Model API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 2️⃣ Read uploaded image
    img = Image.open(file.file).convert("RGB")
    
    # 3️⃣ Resize to model input size (change if your model uses another size)
    img = img.resize((128, 128))
    
    # 4️⃣ Convert to NumPy array and preprocess
    img_array = np.array(img) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # 5️⃣ Make prediction
    prediction = model.predict(img_array)
    result_index = int(np.argmax(prediction))
    model_prediction = class_names[result_index]

    return {
        "class": model_prediction,
        "confidence": float(np.max(prediction))
    }
