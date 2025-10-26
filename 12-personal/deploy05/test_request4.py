# test_request.py
import requests

# URL of your FastAPI server
url = 'http://127.0.0.1:9696/predict'

# JSON data matching your Lead model
client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

# Send POST request
response = requests.post(url, json=client)

# Parse JSON response
prediction = response.json()
print("Response from API:", prediction)

# Optional: print in a friendly way
prob = prediction['conversion_probability']
if prob > 0.5:
    print(f"This lead is likely to convert! Probability: {prob:.2f}")
else:
    print(f"This lead is less likely to convert. Probability: {prob:.2f}")
