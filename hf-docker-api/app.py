from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# Initialize the FastAPI app
app = FastAPI()

# Load the model from Hugging Face
print("Loading the sentiment analysis model...")
model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
sentiment_analyzer = pipeline("sentiment-analysis", model=model_name)
print("Model loaded successfully.")

# Define the input schema
class SentimentRequest(BaseModel):
    text: str

@app.post("/analyze-sentiment")
async def analyze_sentiment(request: SentimentRequest):
    try:
        result = sentiment_analyzer(request.text)
        return {"sentiment": result[0]['label'], "score": result[0]['score']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
