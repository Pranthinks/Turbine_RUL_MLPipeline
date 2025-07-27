from src.Turbine_RUL.logging import logger
from src.Turbine_RUL.components.data_ingestion import DataIngestion
from src.Turbine_RUL.components.data_transformation import DataTransformation
from src.Turbine_RUL.components.feature_engineering import FeatureEngineering

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