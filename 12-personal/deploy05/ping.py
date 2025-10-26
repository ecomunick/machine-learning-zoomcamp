import uvicorn
from fastapi import FastAPI

app = FastAPI(title='Ping') # Create a simple FastAPI application with a ping endpoint

@app.get("/ping") # access the endpoint via url '/ping' (GET request)
def ping():
    return "Pong!"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)

    
