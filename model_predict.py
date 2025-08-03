from src.Turbine_RUL.config.configuration import ConfigurationManager
from src.Turbine_RUL.components.model_prediction import ModelPrediction
from src.Turbine_RUL.logging import logger

STAGE_NAME = "Model Prediction Stage"

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