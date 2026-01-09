from fastapi import FastAPI, UploadFile, File
import shutil
import uuid
import tensorflow as tf
from utils.preprocessing import preprocess_image

app = FastAPI()

model = tf.keras.models.load_model("model/fracture_model.h5")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_file = f"temp_{uuid.uuid4()}.png"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = preprocess_image(temp_file)
    prediction = model.predict(img)[0][0]

    return {
        "result": "Fracture Detected" if prediction > 0.5 else "Normal",
        "confidence": float(prediction)
    }
