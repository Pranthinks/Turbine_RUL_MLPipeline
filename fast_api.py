from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
import uuid
import os
import json
import sys
from datetime import datetime

# Add current directory to Python path
sys.path.append('.')

# Import your model
from src.Turbine_RUL.components.model_prediction import ModelPrediction as BaseModelPrediction

class ModelPrediction(BaseModelPrediction):
    """Enhanced ModelPrediction with NaN handling for API use"""
    
    def prepare_prediction_data(self, test_features):
        """Prepare features for final prediction with NaN handling"""
        
        X_test_features = test_features.reset_index().drop(columns=['unit_id'])
        test_units = test_features.index.to_frame(index=False)
        
        # Get last cycle for each engine
        X_test_last = X_test_features.groupby(test_units['unit_id']).last()
        X_test_last.columns = [str(col) for col in X_test_last.columns]
        engine_ids = X_test_last.index.values
        
        # Check feature alignment
        available_features = set(X_test_last.columns)
        required_features = set(self.selected_features)
        missing_features = required_features - available_features
        
        if missing_features:
            raise ValueError(f"Feature mismatch! Missing: {missing_features}")
        
        X_test_final = X_test_last[self.selected_features]
        
        # Handle NaN values - CRITICAL FIX
        if X_test_final.isnull().sum().sum() > 0:
            # Fill NaN with median of each column
            for col in X_test_final.columns:
                if X_test_final[col].isnull().any():
                    median_val = X_test_final[col].median()
                    if pd.isna(median_val):  # If all values are NaN
                        X_test_final[col] = X_test_final[col].fillna(0)
                    else:
                        X_test_final[col] = X_test_final[col].fillna(median_val)
        
        # Handle infinite values
        X_test_final = X_test_final.replace([np.inf, -np.inf], np.nan)
        # Fill any remaining NaN with 0
        X_test_final = X_test_final.fillna(0)
        
        return X_test_final, engine_ids

app = FastAPI(title="Turbine RUL Prediction API")

@app.post("/predict")
async def predict_rul(file: UploadFile = File(...)):
    """Predict RUL for uploaded turbine test data"""
    
    user_id = str(uuid.uuid4())
    
    try:
        # Create directories
        os.makedirs("temp", exist_ok=True)
        
        # Save uploaded file
        raw_file = f"temp/raw_{user_id}.txt"
        with open(raw_file, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process data to match training format
        processed_file = process_data(raw_file, user_id)
        
        # Run prediction
        predictor = ModelPrediction()
        predictor.config.test_data_path = processed_file
        predictor.config.predictions_path = f"temp/predictions_{user_id}.json"
        
        # Execute prediction pipeline
        predictor.load_artifacts()
        raw_data = pd.read_csv(processed_file)
        features = predictor.preprocess_and_extract_features(raw_data)
        X_test, engine_ids = predictor.prepare_prediction_data(features)
        predictions = predictor.make_predictions(X_test)
        
        # Format results
        results = []
        for engine_id, pred in zip(engine_ids, predictions):
            results.append({
                "engine_id": int(engine_id),
                "predicted_rul": round(float(pred), 1)
            })
        
        # Cleanup
        cleanup_files(user_id)
        
        return {
            "status": "success",
            "total_engines": len(results),
            "predictions": results
        }
        
    except Exception as e:
        cleanup_files(user_id)
        raise HTTPException(status_code=400, detail=str(e))

def process_data(raw_file, user_id):
    """Convert raw user data to EXACT PostgreSQL format like Airflow DAG"""
    
    # STEP 1: Read raw data exactly like Airflow DAG
    df = pd.read_csv(raw_file,
                    sep='\s+',  # Use regex for multiple spaces
                    header=None,
                    skipinitialspace=True)  # Skip leading spaces
    
    # STEP 2: Define column names exactly like Airflow DAG
    columns = ['unit_id', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
              [f'sensor_{i}' for i in range(1, 22)]
    df.columns = columns[:len(df.columns)]  # Match actual column count
    
    # STEP 3: Add RUL column (will be predicted, not provided by user)
    df['rul'] = None  # RUL will be predicted by the model
    
    # STEP 4: Add metadata columns exactly like Airflow DAG
    df['data_type'] = 'test'  # CRITICAL: Must match training data
    df['dataset'] = 'FD001'   # CRITICAL: Match original dataset name
    
    # STEP 5: Add PostgreSQL-specific columns
    df['id'] = range(1, len(df) + 1)  # Auto-increment ID
    df['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    
    # STEP 6: Reorder columns to match PostgreSQL schema EXACTLY
    postgres_columns = [
        'id', 'unit_id', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3',
        'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
        'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
        'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
        'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
        'sensor_21', 'rul', 'data_type', 'dataset', 'created_at'
    ]
    
    # Only include columns that exist in the dataframe
    available_columns = [col for col in postgres_columns if col in df.columns]
    df = df[available_columns]
    
    # STEP 7: Data type consistency (critical for model)
    numeric_columns = ['unit_id', 'time_cycles'] + \
                     [f'op_setting_{i}' for i in range(1, 4)] + \
                     [f'sensor_{i}' for i in range(1, 22)]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # STEP 8: Handle missing values consistently with training
    sensor_cols = [col for col in df.columns if col.startswith(('sensor_', 'op_setting_'))]
    for col in sensor_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)
    
    # STEP 9: Save processed data
    processed_file = f"temp/processed_{user_id}.csv"
    df.to_csv(processed_file, index=False)
    
    return processed_file

def cleanup_files(user_id):
    """Remove temporary files"""
    files_to_remove = [
        f"temp/raw_{user_id}.txt",
        f"temp/processed_{user_id}.csv",
        f"temp/predictions_{user_id}.json"
    ]
    for file_path in files_to_remove:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Turbine RUL Prediction API on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)