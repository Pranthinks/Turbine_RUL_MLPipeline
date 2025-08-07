from src.Turbine_RUL.logging import logger
from src.Turbine_RUL.components.data_ingestion import DataIngestion
from src.Turbine_RUL.components.data_transformation import DataTransformation
from src.Turbine_RUL.components.feature_engineering import FeatureEngineering
from src.Turbine_RUL.components.drift_detection import DriftDetector
from src.Turbine_RUL.components.model_training import ModelTrainer
from src.Turbine_RUL.components.model_prediction import ModelPrediction
from src.Turbine_RUL.config.configuration import ConfigurationManager

class ModelTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_trainer = ModelTrainer()
        model_path, metrics = model_trainer.initiate_model_training()
        
        return model_path, metrics

class ModelPredictionPipeline:
    def __init__(self):
        pass
        
    def main(self):
        config = ConfigurationManager()
        model_prediction_config = config.get_model_prediction_config()
        model_predictor = ModelPrediction()
        
        # The method returns a dictionary, not individual paths
        results = model_predictor.initiate_model_prediction()
        
        # Extract the paths from results dictionary
        predictions_path = results['predictions_path']
        evaluation_metrics_path = results['evaluation_metrics_path']
        
        return predictions_path, evaluation_metrics_path

if __name__ == '__main__':
    # Stage 1: Data Ingestion
    STAGE_NAME = "Data Ingestion stage"
    try:
        logger.info(f"stage {STAGE_NAME} initiated")
        data_ingestion_pipeline = DataIngestion()
        data_ingestion_pipeline.initiate_data_ingestion()
        logger.info(f"Stage {STAGE_NAME} Completed")
    except Exception as e:
        logger.exception(e)
        raise e

    # Stage 2: Data Transformation  
    STAGE_NAME = "Data Transformation stage"
    try:
        logger.info(f"stage {STAGE_NAME} initiated")
        data_transformation_pipeline = DataTransformation()
        data_transformation_pipeline.initiate_data_transformation()
        logger.info(f"Stage {STAGE_NAME} Completed")
    except Exception as e:
        logger.exception(e)
        raise e

    # Stage 3: Feature Engineering
    STAGE_NAME = "Feature Engineering stage"
    try:
        logger.info(f"stage {STAGE_NAME} initiated")
        feature_engineering_pipeline = FeatureEngineering()
        feature_engineering_pipeline.initiate_feature_engineering()
        logger.info(f"Stage {STAGE_NAME} Completed")
    except Exception as e:
        logger.exception(e)
        raise e
    
    #stage 4: Drift Detection
    STAGE_NAME = "Drift Detection"
    try:
        logger.info(">>>>>> Testing Drift Detection <<<<<<")
        detector = DriftDetector()
        drift_detected, report = detector.initiate_drift_detection()
        print(f"\n🎯 FINAL RESULT: Drift Detected = {drift_detected}")
        print(f"📊 Recommendation: {report['recommendation']}")
        if drift_detected:
            print("🔄 Action: Should retrain model")
        else:
            print("⚡ Action: Can proceed with predictions")
            
        logger.info(">>>>>> Drift Detection Test Completed <<<<<<")
        
    except Exception as e:
        logger.exception(e)
        raise e
    
    #Stage 5: Model Training
    STAGE_NAME = "Model Training"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        model_path, metrics = obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
    
    #Stage 6: Model Prediction
    STAGE_NAME = "Model Prediction"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelPredictionPipeline()
        predictions_path, evaluation_metrics_path = obj.main()
        logger.info(f"Predictions saved at: {predictions_path}")
        logger.info(f"Evaluation metrics saved at: {evaluation_metrics_path}")
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e