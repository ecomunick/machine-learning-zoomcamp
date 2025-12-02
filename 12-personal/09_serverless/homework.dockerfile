FROM agrigorev/model-2025-hairstyle:v1

# WORKDIR /app
WORKDIR /var/task

RUN python -m pip install --no-cache-dir pillow numpy requests onnxruntime

COPY predict_image.py ./
COPY lambda_function.py ./

CMD ["lambda_function.lambda_handler"]



