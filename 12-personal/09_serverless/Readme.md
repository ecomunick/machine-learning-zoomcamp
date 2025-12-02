Module 9 – Serverless Hair Classifier (Docker + AWS Lambda Local)
1. Prepare Project Folder

Create a folder for the homework:

```
mkdir 09_serverless
cd 09_serverless
```

2. Create predict_image.py
'''
import onnxruntime as ort
import numpy as np
from PIL import Image
import requests
from io import BytesIO
'''

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
    return float(outputs[0][0][0])

def lambda_handler(event, context):
    url = event.get("url")
    if not url:
        return {"error": "No URL provided"}
    pred = predict_from_url(url)
    return {"prediction": pred}

3. (Optional) Create lambda_function.py

This is optional if you want a separate handler:

from predict_image import predict_from_url

def lambda_handler(event, context):
    url = event.get("url")
    pred = predict_from_url(url)
    return {"prediction": float(pred)}

4. Create Dockerfile (homework.dockerfile)
FROM agrigorev/model-2025-hairstyle:v1

# AWS Lambda expects /var/task as WORKDIR
WORKDIR /var/task

# Install Python dependencies
RUN python -m pip install --no-cache-dir pillow numpy requests onnxruntime

# Copy code into container
COPY predict_image.py ./
COPY lambda_function.py ./  # optional

# Lambda entry point (match the file + function)
CMD ["predict_image.lambda_handler"]

5. Place ONNX model

Make sure the ONNX model hair_classifier_empty.onnx is in the same folder:

ls -l
# should show:
# hair_classifier_empty.onnx
# predict_image.py
# lambda_function.py
# homework.dockerfile

6. Build Docker Image
DOCKER_BUILDKIT=0 docker build \
  --platform linux/amd64 \
  -t my-hairstyle-lambda \
  -f homework.dockerfile .


Ignore platform warnings if on M1/M2 Mac.

Make sure all dependencies install successfully.

7. Inspect Container (Optional)

Check that files are copied correctly:

docker run --rm -it --platform linux/amd64 --entrypoint sh my-hairstyle-lambda
ls -l /var/task
# should show:
# predict_image.py
# lambda_function.py (if copied)

8. Run Lambda Locally
docker run --rm -p 9000:8080 --platform linux/amd64 my-hairstyle-lambda


Container will run the Lambda runtime locally on port 9000.

9. Invoke Lambda with curl

Open another terminal and run:

curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
-H "Content-Type: application/json" \
-d '{"url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"}'


Expected output:

{"prediction": -0.017499880865216255}


For multiple-choice questions, closest answer: -0.10

10. Troubleshooting / Common Issues

Unable to import module lambda_function → fix by matching CMD ["predict_image.lambda_handler"] in Dockerfile and ensure the file exists in /var/task.

Files not found → ensure COPY commands point to correct host files.

Platform warnings → safe to ignore if container runs.

Inspect container:

docker run --rm -it --platform linux/amd64 --entrypoint sh my-hairstyle-lambda
ls -l /var/task


Lambda runtime requires /var/task as working directory. That was the main reason your earlier builds failed.

11. Optional: Local Test Script
from predict_image import predict_from_url

url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
print(predict_from_url(url))


Useful to test ONNX inference without Docker.