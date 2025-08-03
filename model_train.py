from src.Turbine_RUL.config.configuration import ConfigurationManager
from src.Turbine_RUL.components.model_training import ModelTrainer
from src.Turbine_RUL.logging import logger


STAGE_NAME = "Model Training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_trainer = ModelTrainer()
        model_path, metrics = model_trainer.initiate_model_training()
        
        return model_path, metrics

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        model_path, metrics = obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e