text
---
title: "Module 9 – Serverless Hair Classifier (Docker + AWS Lambda Local)"
description: "Complete guide for deploying hair classifier model using Docker and AWS Lambda locally"
date: 2025-12-02
category: serverless
tags: ["docker", "aws-lambda", "onnx", "computer-vision"]
---

# Module 9 – Serverless Hair Classifier (Docker + AWS Lambda Local)

## 1. Prepare Project Folder

Create a folder for the homework:

mkdir 09_serverless
cd 09_serverless

text

## 2. Create predict_image.py

<details>
<summary>Paste the following code</summary>

import onnxruntime as ort
import numpy as np
from PIL import Image
import requests
from io import BytesIO

Load the model once
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
inputs = {session.get_inputs().name: x}
outputs = session.run(None, inputs)
return float(outputs)

def lambda_handler(event, context):
url = event.get("url")
if not url:
return {"error": "No URL provided"}
pred = predict_from_url(url)
return {"prediction": pred}

text

</details>

## 3. (Optional) Create lambda_function.py

<details>
<summary>Optional separate handler code</summary>

from predict_image import predict_from_url

def lambda_handler(event, context):
url = event.get("url")
pred = predict_from_url(url)
return {"prediction": float(pred)}

text

</details>

## 4. Create Dockerfile (homework.dockerfile)

<details>
<summary>Dockerfile content</summary>

FROM agrigorev/model-2025-hairstyle:v1

AWS Lambda expects /var/task as WORKDIR
WORKDIR /var/task

Install Python dependencies
RUN python -m pip install --no-cache-dir pillow numpy requests onnxruntime

Copy code into container
COPY predict_image.py ./
COPY lambda_function.py ./ # optional

Lambda entry point (match the file + function)
CMD ["predict_image.lambda_handler"]

text

</details>

## 5. Place ONNX model

Make sure the ONNX model `hair_classifier_empty.onnx` is in the same folder:

ls -l

text

Expected files:
hair_classifier_empty.onnx
predict_image.py
lambda_function.py
homework.dockerfile

text

## 6. Build Docker Image

DOCKER_BUILDKIT=0 docker build
--platform linux/amd64
-t my-hairstyle-lambda
-f homework.dockerfile .

text

**Note**: Ignore platform warnings on M1/M2 Mac. Ensure dependencies install successfully.[web:28]

## 7. Inspect Container (Optional)

docker run --rm -it --platform linux/amd64 --entrypoint sh my-hairstyle-lambda
ls -l /var/task

text

## 8. Run Lambda Locally

docker run --rm -p 9000:8080 --platform linux/amd64 my-hairstyle-lambda

text

## 9. Invoke Lambda with curl

New terminal:

curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations"
-H "Content-Type: application/json"
-d '{"url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"}'

text

Expected: `{"prediction": -0.017499880865216255}` (closest: -0.10)[web:3]

## 10. Troubleshooting

- **Unable to import module**: Match `CMD ["predict_image.lambda_handler"]` in Dockerfile
- **Files not found**: Verify COPY commands point to correct files
- **Platform warnings**: Safe to ignore if container runs

Lambda requires `/var/task` working directory.[web:28]

## 11. Optional: Local Test Script

<details>
<summary>Local test without Docker</summary>

from predict_image import predict_from_url

url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
print(predict_from_url(url))

text

</details>