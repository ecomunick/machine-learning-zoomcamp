from predict_image import predict_from_url

def lambda_handler(event, context):
    url = event.get("url")
    pred = predict_from_url(url)
    return {"prediction": float(pred)}
