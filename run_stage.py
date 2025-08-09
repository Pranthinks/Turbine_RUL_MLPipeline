#!/usr/bin/env python3
import sys
from src.Turbine_RUL.logging import logger
from src.Turbine_RUL.components.data_ingestion import DataIngestion
from src.Turbine_RUL.components.data_transformation import DataTransformation
from src.Turbine_RUL.components.feature_engineering import FeatureEngineering
from src.Turbine_RUL.components.drift_detection import DriftDetector
from main import ModelTrainingPipeline, ModelPredictionPipeline

def run_stage(stage_name):
    """Run individual pipeline stages"""
        
    if stage_name == "1" or stage_name == "data_ingestion":
        logger.info("Starting Data Ingestion")
        DataIngestion().initiate_data_ingestion()
            
    elif stage_name == "2" or stage_name == "drift_detection":
        logger.info("Starting Drift Detection")
        detector = DriftDetector()
        drift_detected, report = detector.initiate_drift_detection()
        print(f"\n🎯 FINAL RESULT: Drift Detected = {drift_detected}")
        print(f"📊 Recommendation: {report['recommendation']}")
        if drift_detected:
            print("🔄 Action: Should retrain model")
        else:
            print("⚡ Action: Can proceed with predictions")
            
    elif stage_name == "3" or stage_name == "data_transformation":
        logger.info("Starting Data Transformation")
        DataTransformation().initiate_data_transformation()
            
    elif stage_name == "4" or stage_name == "feature_engineering":
        logger.info("Starting Feature Engineering")
        FeatureEngineering().initiate_feature_engineering()
            
    elif stage_name == "5" or stage_name == "model_training":
        logger.info("Starting Model Training")
        ModelTrainingPipeline().main()
            
    elif stage_name == "6" or stage_name == "model_prediction":
        logger.info("Starting Model Prediction")
        ModelPredictionPipeline().main()
        
    elif stage_name == "7" or stage_name == "calculate_reference":
        logger.info("Starting Reference Data Calculation")
        print("🔧 CALCULATING REFERENCE DATA FOR DRIFT DETECTION")
        print("="*60)
        detector = DriftDetector()
        reference_stats = detector.calculate_and_save_reference()
        print("\n✅ Reference data calculation completed!")
        print("💡 You can now run drift detection (stage 2) in future runs")
        print("="*60)
            
    else:
        print("Available stages:")
        print("1 or data_ingestion")
        print("2 or drift_detection")
        print("3 or data_transformation") 
        print("4 or feature_engineering")
        print("5 or model_training")
        print("6 or model_prediction")
        print("7 or calculate_reference  <- Run this ONCE to setup reference data")
        return
    
    logger.info(f"Stage {stage_name} completed successfully!")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python run_stage.py <stage_number_or_name>")
        print("\nExamples:")
        print("python run_stage.py 1")
        print("python run_stage.py data_ingestion")
        print("python run_stage.py 7  # Calculate reference data (run once)")
        sys.exit(1)
        
    stage = sys.argv[1]
    run_stage(stage)