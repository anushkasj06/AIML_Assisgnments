from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Initialize FastAPI app
app = FastAPI(title="LSTM Text Predictor API", description="Predicts the next word in a sequence.")

# Global variables for model and tokenizer
model = None
tokenizer = None
max_seq_len = None

# Input Data Schema
class TextRequest(BaseModel):
    text: str

# Load artifacts on startup
@app.on_event("startup")
def load_artifacts():
    global model, tokenizer, max_seq_len
    try:
        model = tf.keras.models.load_model('lstm_next_word_model.h5')
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open('max_seq_len.txt', 'r') as f:
            max_seq_len = int(f.read().strip())
        print("Model and artifacts loaded successfully.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

# Create predict endpoint
@app.post("/predict")
def predict_next_word(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    # Preprocess the input text
    token_list = tokenizer.texts_to_sequences([request.text])[0]
    # Pad the sequence
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
    
    # Generate prediction
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probs, axis=-1)[0]
    
    # Map index back to word
    predicted_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            predicted_word = word
            break
            
    return {
        "input_text": request.text,
        "predicted_next_word": predicted_word,
        "complete_text": f"{request.text} {predicted_word}"
    }

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the LSTM Text Prediction API. Use the /predict endpoint."}
