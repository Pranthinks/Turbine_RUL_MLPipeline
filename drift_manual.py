from src.Turbine_RUL.config.configuration import ConfigurationManager
from src.Turbine_RUL.components.drift_detection import DriftDetector
from src.Turbine_RUL.logging import logger

STAGE_NAME = "Manual Drift Reference Calculation"

class ManualDriftReferencePipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        drift_detection_config = config.get_drift_detection_config()
        drift_detector = DriftDetector()
        
        # Calculate and save reference data
        reference_stats = drift_detector.calculate_and_save_reference()
        
        return reference_stats

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ManualDriftReferencePipeline()
        reference_stats = obj.main()
        logger.info(f"Reference calculation completed for {len(reference_stats)} features")
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e