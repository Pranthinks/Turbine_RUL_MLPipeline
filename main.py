from src.Turbine_RUL.logging import logger
from src.Turbine_RUL.components.data_ingestion import DataIngestion


STAGE_NAME = "Data Ingestion stage"


try:
    logger.info(f"stage {STAGE_NAME} initiated")
    data_ingestion_pipeline = DataIngestion()
    data_ingestion_pipeline.initiate_data_ingestion()
    logger.info(f"Stage {STAGE_NAME} Completed")
except Exception as e:
    logger.exception(e)
    raise e
