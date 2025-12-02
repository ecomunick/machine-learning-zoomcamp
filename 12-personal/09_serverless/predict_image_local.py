# predict_image.py
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image
import onnxruntime as ort

MODEL_FILE = "hair_classifier_v1.onnx"      # or hair_classifier_empty.onnx in the Docker image
IMAGE_URL = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess(img):
    # The course used: convert to numpy, transpose to CHW, scale to [0,1], then normalize to [-1,1]
    # i.e. x = (x / 255.0 - 0.5) / 0.5 -> x*2 -1
    arr = np.array(img).astype(np.float32)  # H,W,C
    # transpose to C,H,W
    arr = arr.transpose(2,0,1)
    # scale to [0,1]
    arr /= 255.0
    # normalize to [-1,1]
    arr = (arr - 0.5) / 0.5
    return arr

def print_first_pixel_r(arr):
    # arr is CHW, first pixel is [C, H=0, W=0]
    r = arr[0,0,0]
    print("First pixel R after preprocessing:", r)

def run_model(model_file, inp):
    sess = ort.InferenceSession(model_file, providers=['CPUExecutionProvider'])
    # find input name
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print("Using input:", input_name, "output:", output_name)

    # model expects batch dimension
    inp_batch = np.expand_dims(inp, axis=0).astype(np.float32)
    res = sess.run(None, {input_name: inp_batch})
    print("Raw model outputs:", res)
    # If model output is single value (logit), print that value
    if isinstance(res, list):
        if len(res) == 1:
            val = res[0]
            # val could be shape (1,1) or (1,)
            print("Model output value:", val)
        else:
            print("Model returned multiple arrays.")
    else:
        print("Model returned:", res)

def main():
    img = download_image(IMAGE_URL)
    # target size — set here to one of: (64,64), (128,128), etc.
    # target_size = (64, 64)   # <--- change if you used a different target size
    target_size = (200, 200)
    img_pre = prepare_image(img, target_size)
    arr = preprocess(img_pre)
    print_first_pixel_r(arr)
    run_model(MODEL_FILE, arr)

if __name__ == "__main__":
    main()
