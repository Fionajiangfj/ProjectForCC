from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# Initialize the FastAPI app
app = FastAPI()

# Load the model from Hugging Face
print("Loading the sentiment analysis model...")
pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
print("Model loaded successfully.")

@app.get("/")
def read_root():
    return {"message": "API is up and running!"}

# Define the input schema
class SentimentRequest(BaseModel):
    inputs: str

@app.post("/predict")
async def analyze_sentiment(request: SentimentRequest):
    try:
        result = pipe(request.inputs)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
