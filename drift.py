from src.Turbine_RUL.components.drift_detection import DriftDetector
from src.Turbine_RUL.logging import logger

if __name__ == '__main__':
    try:
        logger.info(">>>>>> Testing Drift Detection <<<<<<")
        
        # Create drift detector
        detector = DriftDetector()
        
        # Run drift detection (it will use test data from config)
        drift_detected, report = detector.initiate_drift_detection()
        
        # Print results
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