from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import json
import requests

app = FastAPI()

entry_points = [
    "http://localhost",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = entry_points,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],      
    
)

end_point = "http://localhost:8501/v1/models/tomatoes_model:predict"

#MODEL = tf.keras.models.load_model("../models/1")
MODEL = tf.keras.models.load_model("../tomatoes.h5")

with open("../classNames.json") as f:
    CLASS_NAMES = json.loads(f.read())
    
@app.get("/home")
async def welcome_home():
    return "Connected to the the server ..."

def read_file_as_image(data) -> np.ndarray:
    return np.array(Image.open(BytesIO(data)))

@app.post("/predict")
async def predict_cls( file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    json_image = {
        "instances": img_batch.tolist()
    }
       
    output = requests.post(end_point, json=json_image)
    prediction = np.array(output.json()["predictions"][0])
    
    return {
        'class': CLASS_NAMES[np.argmax(prediction)],
        'confidence': round(100 * (np.max(prediction)), 2)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=5000)
    


 