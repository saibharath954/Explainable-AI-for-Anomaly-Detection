from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import logging
from typing import List, Dict, Any

# Import our modules
from .preprocess import Preprocessor
from .detector import IsfDetector
from .explainer import SHAPExplainer
from .config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, MODELS_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description="API for detecting fraudulent transactions using Isolation Forest",
    version="1.0.0"
)

# Load models at startup
@app.on_event("startup")
async def load_models():
    try:
        global preprocessor, detector, explainer
        
        logger.info("Loading preprocessor...")
        preprocessor = Preprocessor.load()
        
        logger.info("Loading detector...")
        detector = IsfDetector.load()
        
        logger.info("Loading explainer...")
        explainer = SHAPExplainer.load()
        
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise e

# Pydantic models for request/response
class Transaction(BaseModel):
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

class PredictionResponse(BaseModel):
    is_fraud: bool
    anomaly_score: float
    threshold: float
    explanation: str
    top_features: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

@app.get("/")
async def root():
    return {"message": "Fraud Detection API"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=hasattr(detector, 'is_fitted_') and detector.is_fitted_
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction):
    try:
        # Convert to DataFrame
        transaction_dict = transaction.dict()
        df = pd.DataFrame([transaction_dict])
        
        # Preprocess
        X = preprocessor.transform(df)
        
        # Predict
        prediction, score = detector.predict(X)
        is_fraud = bool(prediction[0])
        anomaly_score = float(score[0])
        
        # Generate explanation if it's fraud
        explanation = ""
        top_features = []
        
        if is_fraud:
            # Get SHAP values
            shap_values = explainer.explain_instance(X[0])
            
            # Generate natural language explanation
            feature_names = preprocessor.get_feature_names_out()
            explanation = explainer.generate_natural_language_explanation(
                shap_values, X[0], feature_names
            )
            
            # Get top contributing features
            abs_shap = np.abs(shap_values)
            top_indices = np.argsort(-abs_shap)[:3]  # Top 3 features
            
            for idx in top_indices:
                feature_name = feature_names[idx]
                shap_value = float(shap_values[idx])
                feature_value = float(X[0][idx])
                
                top_features.append({
                    "feature": feature_name,
                    "value": feature_value,
                    "shap_contribution": shap_value,
                    "direction": "increases" if shap_value > 0 else "decreases"
                })
        
        return PredictionResponse(
            is_fraud=is_fraud,
            anomaly_score=anomaly_score,
            threshold=float(detector.threshold_),
            explanation=explanation,
            top_features=top_features
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(transactions: List[Transaction]):
    try:
        # Convert to DataFrame
        transactions_dict = [t.dict() for t in transactions]
        df = pd.DataFrame(transactions_dict)
        
        # Preprocess
        X = preprocessor.transform(df)
        
        # Predict
        predictions, scores = detector.predict(X)
        
        results = []
        for i, (pred, score) in enumerate(zip(predictions, scores)):
            results.append({
                "transaction_id": i,
                "is_fraud": bool(pred),
                "anomaly_score": float(score),
                "above_threshold": bool(score >= detector.threshold_)
            })
        
        return {
            "predictions": results,
            "threshold": float(detector.threshold_)
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)