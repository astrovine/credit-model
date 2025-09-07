from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List
import json

print("Loading fraud detection model...")

try:
    model = joblib.load('fraud_detection_model.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    
    # Load model metadata
    with open('model_metadata.json', 'r') as f:
        model_info = json.load(f)
    
    print("✓ Model loaded successfully!")
    print(f"✓ Model accuracy: {model_info['test_accuracy']:.4f}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="ML-powered API to detect fraudulent credit card transactions",
    version="1.0.0"
)

class Transaction(BaseModel):
    """
    Input model for a credit card transaction
    """
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    
    class Config:
        # Example transaction for API documentation
        schema_extra = {
            "example": {
                "Time": 144113.0,
                "V1": -0.5,
                "V2": 0.5,
                "V3": 1.2,
                "V4": -0.8,
                "V5": 0.3,
                "V6": -1.1,
                "V7": 0.9,
                "V8": -0.4,
                "V9": 0.7,
                "V10": -0.6,
                "V11": 1.3,
                "V12": -0.9,
                "V13": 0.2,
                "V14": -1.5,
                "V15": 0.8,
                "V16": -0.3,
                "V17": 1.0,
                "V18": -0.7,
                "V19": 0.4,
                "V20": -1.2,
                "V21": 0.6,
                "V22": -0.1,
                "V23": 0.9,
                "V24": -0.5,
                "V25": 0.2,
                "V26": -0.8,
                "V27": 1.1,
                "V28": -0.4,
                "Amount": 124.00
            }
        }

class PredictionResponse(BaseModel):
    """
    Output model for fraud prediction
    """
    is_fraud: bool
    fraud_probability: float
    risk_score: str
    confidence: float
    model_version: str

@app.get("/")
async def root():
    """
    Health check endpoint
    """
    return {
        "message": "Credit Card Fraud Detection API",
        "status": "healthy",
        "model_accuracy": model_info["test_accuracy"],
        "model_type": model_info["model_type"]
    }

@app.get("/model-info")
async def get_model_info():
    """
    Get information about the trained model
    """
    return model_info

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: Transaction):
    """
    Predict if a transaction is fraudulent
    
    Args:
        transaction: Credit card transaction data
        
    Returns:
        Fraud prediction with probability and risk assessment
    """
    try:
        # Convert transaction to dataframe
        transaction_dict = transaction.dict()
        df_transaction = pd.DataFrame([transaction_dict])
        
        # Scale features using the same scaler from training
        scaled_features = scaler.transform(df_transaction)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0]
        
        # Calculate fraud probability and risk score
        fraud_prob = probability[1]  # Probability of fraud (class 1)
        normal_prob = probability[0]  # Probability of normal (class 0)
        
        # Determine risk level
        if fraud_prob >= 0.7:
            risk_level = "HIGH"
        elif fraud_prob >= 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return PredictionResponse(
            is_fraud=bool(prediction == 1),
            fraud_probability=float(fraud_prob),
            risk_score=risk_level,
            confidence=float(max(normal_prob, fraud_prob)),
            model_version="1.0.0"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(transactions: List[Transaction]):
    """
    Predict fraud for multiple transactions
    
    Args:
        transactions: List of credit card transactions
        
    Returns:
        List of fraud predictions
    """
    try:
        results = []
        
        for transaction in transactions:
            # Reuse the single prediction logic
            result = await predict_fraud(transaction)
            results.append(result)
        
        return {
            "predictions": results,
            "total_transactions": len(transactions),
            "fraud_detected": sum(1 for r in results if r.is_fraud)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("API Documentation available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
