import onnxruntime as ort
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Load the model once
session = ort.InferenceSession("hair_classifier_empty.onnx", providers=["CPUExecutionProvider"])

def preprocess(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image = image.resize((200, 200))
    img_arr = np.array(image).astype("float32") / 255.0
    img_arr = np.transpose(img_arr, (2, 0, 1))
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

def predict_from_url(url):
    x = preprocess(url)
    inputs = {session.get_inputs()[0].name: x}
    outputs = session.run(None, inputs)
    pred = float(outputs[0][0][0])
    return pred

def lambda_handler(event, context):
    url = event.get("url")
    if not url:
        return {"error": "No URL provided"}

    x = preprocess(url)
    inputs = {session.get_inputs()[0].name: x}
    outputs = session.run(None, inputs)
    pred = float(outputs[0][0][0])
    return {"prediction": pred}
