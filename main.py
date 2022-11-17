from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
""" import uvicorn """
from waitress import serve
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import json
import os

app = FastAPI()

entry_points = [
    "http://localhost",
    "http://localhost:3000",
    "https://tomato-disease-clsfn.herokuapp.com/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = entry_points,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],      
    
)

#MODEL = tf.keras.models.load_model("../models/1")
MODEL = tf.keras.models.load_model("./artifacts/tomatoes.h5")

with open("./artifacts/classNames.json") as f:
    CLASS_NAMES = json.loads(f.read())
CLASS_NAMES = [nm.split("Tomato__")[-1] if nm.count("Tomato")>1 else nm for nm in CLASS_NAMES]
#CLASS_NAMES = [nm.replace("__", " - ") if nm.count("__") else nm for nm in CLASS_NAMES]
    
@app.get("/home")
async def welcome_home():
    return "Connected to the the server ..."

def read_file_as_image(data) -> np.ndarray:
    return np.array(Image.open(BytesIO(data)))

def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

@app.post("/predict")
async def predict_cls( file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
       
    predict_cls, confidence = predict(MODEL, image)
    
    return {
        'class': predict_cls,
        'confidence': confidence,
    }
@app.get("/classes")
async def disease_cls( ):
    
    return list(CLASS_NAMES)

PORT = os.environ.get('PORT', 5000)

if __name__ == "__main__":
    """ uvicorn.run(app, port=PORT) """
    serve(app, port=PORT)
    


 