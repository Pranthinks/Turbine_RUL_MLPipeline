# src/Turbine_RUL/monitoring/enhanced_metrics.py
# Enhanced monitoring for your 5-stage MLOps pipeline

import os
import time
import psutil
from prometheus_client import Gauge, Counter, Histogram, push_to_gateway, CollectorRegistry

class TurbineMLOpsMetrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # Get Pushgateway URL from environment
        self.pushgateway_url = os.getenv('PUSHGATEWAY_URL', 'localhost:9091')
        self.monitoring_enabled = os.getenv('PROMETHEUS_ENABLED', 'false').lower() == 'true'
        
        if self.monitoring_enabled:
            print(f"✅ Enhanced Monitoring enabled - Pushgateway: {self.pushgateway_url}")
        else:
            print("⚠️ Monitoring disabled")
        
        # 1. PIPELINE PERFORMANCE METRICS
        self.stage_duration = Histogram(
            'pipeline_stage_duration_seconds',
            'Time taken for each pipeline stage',
            ['stage_name', 'status'],
            buckets=[1, 5, 10, 30, 60, 300, 600, 1800, 3600, float('inf')],
            registry=self.registry
        )
        
        self.stage_counter = Counter(
            'pipeline_stage_executions_total',
            'Total executions of each stage',
            ['stage_name', 'status'],
            registry=self.registry
        )
        
        # 2. DATA QUALITY METRICS
        self.data_quality_score = Gauge(
            'data_quality_score',
            'Data quality score (0-1)',
            ['dataset_type', 'stage'],
            registry=self.registry
        )
        
        self.dataset_size = Gauge(
            'dataset_size_records',
            'Number of records in dataset',
            ['dataset_type', 'stage'],
            registry=self.registry
        )
        
        self.dataset_features = Gauge(
            'dataset_features_count',
            'Number of features in dataset',
            ['dataset_type', 'stage'],
            registry=self.registry
        )
        
        self.missing_values_percentage = Gauge(
            'missing_values_percentage',
            'Percentage of missing values',
            ['dataset_type', 'column_type'],
            registry=self.registry
        )
        
        # 3. DRIFT DETECTION METRICS
        self.drift_detected = Gauge(
            'drift_detected',
            'Whether drift was detected (1=yes, 0=no)',
            ['drift_type'],
            registry=self.registry
        )
        
        self.psi_score = Gauge(
            'psi_score',
            'Population Stability Index score',
            ['feature'],
            registry=self.registry
        )
        
        self.drift_features_count = Gauge(
            'drift_features_count',
            'Number of features with detected drift',
            registry=self.registry
        )
        
        self.drift_severity = Gauge(
            'drift_severity_score',
            'Overall drift severity (0-1)',
            registry=self.registry
        )
        
        # 4. DATA TRANSFORMATION METRICS
        self.transformation_data_reduction = Gauge(
            'transformation_data_reduction_ratio',
            'Ratio of data reduction during transformation',
            registry=self.registry
        )
        
        self.features_dropped_count = Gauge(
            'features_dropped_count',
            'Number of features dropped during transformation',
            ['drop_reason'],
            registry=self.registry
        )
        
        # 5. FEATURE ENGINEERING METRICS
        self.feature_extraction_time = Gauge(
            'feature_extraction_duration_seconds',
            'Time taken for feature extraction',
            ['feature_type'],
            registry=self.registry
        )
        
        self.engineered_features_count = Gauge(
            'engineered_features_count',
            'Number of engineered features created',
            ['feature_type'],
            registry=self.registry
        )
        
        self.feature_selection_ratio = Gauge(
            'feature_selection_ratio',
            'Ratio of features selected vs total',
            registry=self.registry
        )
        
        # 6. MODEL TRAINING METRICS
        self.model_performance_rmse = Gauge(
            'model_rmse_score',
            'Model RMSE performance',
            ['split_type', 'model_type'],
            registry=self.registry
        )
        
        self.model_performance_mae = Gauge(
            'model_mae_score',
            'Model MAE performance',
            ['split_type', 'model_type'],
            registry=self.registry
        )
        
        self.training_samples_count = Gauge(
            'training_samples_count',
            'Number of training samples used',
            registry=self.registry
        )
        
        self.hyperparameter_trials = Counter(
            'hyperparameter_trials_total',
            'Total hyperparameter optimization trials',
            ['model_type'],
            registry=self.registry
        )
        
        self.best_model_improvement = Gauge(
            'best_model_improvement_percentage',
            'Improvement of best model over baseline',
            registry=self.registry
        )
        
        # 7. MODEL PREDICTION METRICS
        self.prediction_latency = Histogram(
            'prediction_latency_seconds',
            'Prediction latency per sample',
            buckets=[0.001, 0.01, 0.1, 1, 5, 10, float('inf')],
            registry=self.registry
        )
        
        self.prediction_count = Counter(
            'predictions_total',
            'Total number of predictions made',
            registry=self.registry
        )
        
        self.rul_prediction_distribution = Histogram(
            'rul_prediction_values',
            'Distribution of RUL prediction values',
            buckets=[0, 10, 25, 50, 75, 100, 135, 200, 300, float('inf')],
            registry=self.registry
        )
        
        self.prediction_accuracy_metrics = Gauge(
            'prediction_accuracy_score',
            'Prediction accuracy metrics',
            ['metric_type'],
            registry=self.registry
        )
        
        # 8. SYSTEM RESOURCE METRICS
        self.memory_usage_gb = Gauge(
            'system_memory_usage_gb',
            'System memory usage in GB',
            registry=self.registry
        )
        
        self.cpu_usage_percent = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.disk_usage_percent = Gauge(
            'system_disk_usage_percent',
            'System disk usage percentage',
            ['mount_point'],
            registry=self.registry
        )
        
        # 9. BUSINESS METRICS
        self.pipeline_sla_compliance = Gauge(
            'pipeline_sla_compliance',
            'Pipeline SLA compliance (1=compliant, 0=breach)',
            registry=self.registry
        )
        
        self.total_pipeline_duration = Gauge(
            'total_pipeline_duration_seconds',
            'End-to-end pipeline execution time',
            registry=self.registry
        )
        
        self.artifact_size_mb = Gauge(
            'artifact_size_mb',
            'Size of pipeline artifacts in MB',
            ['artifact_type'],
            registry=self.registry
        )
    
    def push_metrics(self, job_name='turbine_rul_pipeline'):
        """Push metrics to Pushgateway"""
        if not self.monitoring_enabled:
            return
            
        try:
            push_to_gateway(
                self.pushgateway_url, 
                job=job_name, 
                registry=self.registry
            )
            print(f"📊 Enhanced metrics pushed to {self.pushgateway_url}")
        except Exception as e:
            print(f"⚠️ Failed to push metrics: {e}")
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        if not self.monitoring_enabled:
            return
            
        # Memory usage
        memory_gb = psutil.virtual_memory().used / (1024**3)
        self.memory_usage_gb.set(memory_gb)
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.cpu_usage_percent.set(cpu_percent)
        
        # Disk usage
        try:
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            self.disk_usage_percent.labels(mount_point='/').set(disk_percent)
        except:
            pass  # Windows compatibility
    
    def record_stage_completion(self, stage_name, duration, status='success', data_shape=None):
        """Record stage completion with enhanced metrics"""
        if not self.monitoring_enabled:
            return
            
        self.stage_duration.labels(stage_name=stage_name, status=status).observe(duration)
        self.stage_counter.labels(stage_name=stage_name, status=status).inc()
        
        if data_shape:
            self.dataset_size.labels(dataset_type='processed', stage=stage_name).set(data_shape[0])
            if len(data_shape) > 1:
                self.dataset_features.labels(dataset_type='processed', stage=stage_name).set(data_shape[1])
        
        print(f"📈 Enhanced metrics recorded for {stage_name}: {duration:.2f}s ({status})")
    
    def record_data_ingestion_metrics(self, train_df, test_df):
        """Record comprehensive data ingestion metrics"""
        if not self.monitoring_enabled:
            return
            
        # Dataset sizes
        self.dataset_size.labels(dataset_type='train', stage='ingestion').set(len(train_df))
        self.dataset_size.labels(dataset_type='test', stage='ingestion').set(len(test_df))
        
        # Feature counts
        self.dataset_features.labels(dataset_type='train', stage='ingestion').set(train_df.shape[1])
        self.dataset_features.labels(dataset_type='test', stage='ingestion').set(test_df.shape[1])
        
        # Data quality scores
        train_quality = 1 - (train_df.isnull().sum().sum() / (train_df.shape[0] * train_df.shape[1]))
        test_quality = 1 - (test_df.isnull().sum().sum() / (test_df.shape[0] * test_df.shape[1]))
        
        self.data_quality_score.labels(dataset_type='train', stage='ingestion').set(train_quality)
        self.data_quality_score.labels(dataset_type='test', stage='ingestion').set(test_quality)
        
        # Missing values by sensor type
        sensor_cols = [col for col in train_df.columns if col.startswith('sensor')]
        if sensor_cols:
            sensor_missing_pct = (train_df[sensor_cols].isnull().sum().sum() / 
                                (len(train_df) * len(sensor_cols))) * 100
            self.missing_values_percentage.labels(dataset_type='train', column_type='sensor').set(sensor_missing_pct)
    
    def record_drift_detection_metrics(self, drift_report):
        """Record comprehensive drift detection metrics"""
        if not self.monitoring_enabled:
            return
            
        # Overall drift flags
        self.drift_detected.labels(drift_type='overall').set(1 if drift_report.get('overall_drift_detected', False) else 0)
        self.drift_detected.labels(drift_type='data').set(1 if drift_report.get('data_drift_detected', False) else 0)
        self.drift_detected.labels(drift_type='quality').set(1 if drift_report.get('quality_drift_detected', False) else 0)
        self.drift_detected.labels(drift_type='temporal').set(1 if drift_report.get('temporal_drift_detected', False) else 0)
        
        # Feature-level drift metrics
        feature_drifts = drift_report.get('feature_drifts', {})
        drift_count = 0
        total_psi = 0
        
        for feature, drift_info in feature_drifts.items():
            if 'psi_score' in drift_info:
                self.psi_score.labels(feature=feature).set(drift_info['psi_score'])
                total_psi += drift_info['psi_score']
            if drift_info.get('drift_detected', False):
                drift_count += 1
        
        self.drift_features_count.set(drift_count)
        
        # Drift severity (average PSI score)
        if len(feature_drifts) > 0:
            avg_psi = total_psi / len(feature_drifts)
            self.drift_severity.set(min(avg_psi, 1.0))  # Cap at 1.0
    
    def record_transformation_metrics(self, original_shape, transformed_shape, dropped_features=None):
        """Record data transformation metrics"""
        if not self.monitoring_enabled:
            return
            
        # Data reduction ratio
        reduction_ratio = transformed_shape[0] / original_shape[0] if original_shape[0] > 0 else 1
        self.transformation_data_reduction.set(reduction_ratio)
        
        # Features dropped
        if dropped_features:
            for reason, count in dropped_features.items():
                self.features_dropped_count.labels(drop_reason=reason).set(count)
    
    def record_feature_engineering_metrics(self, extraction_times, feature_counts, selection_ratio):
        """Record feature engineering metrics"""
        if not self.monitoring_enabled:
            return
            
        # Feature extraction times
        for feature_type, duration in extraction_times.items():
            self.feature_extraction_time.labels(feature_type=feature_type).set(duration)
        
        # Feature counts
        for feature_type, count in feature_counts.items():
            self.engineered_features_count.labels(feature_type=feature_type).set(count)
        
        # Selection ratio
        self.feature_selection_ratio.set(selection_ratio)
    
    def record_training_metrics(self, cv_results, best_model_name, n_samples, hyperparameter_trials=0):
        """Record comprehensive model training metrics"""
        if not self.monitoring_enabled:
            return
            
        # Model performance
        if 'test_rmse' in cv_results.columns:
            self.model_performance_rmse.labels(split_type='test', model_type=best_model_name).set(cv_results['test_rmse'].mean())
            self.model_performance_rmse.labels(split_type='train', model_type=best_model_name).set(cv_results['train_rmse'].mean())
        
        if 'test_mae' in cv_results.columns:
            self.model_performance_mae.labels(split_type='test', model_type=best_model_name).set(cv_results['test_mae'].mean())
            self.model_performance_mae.labels(split_type='train', model_type=best_model_name).set(cv_results['train_mae'].mean())
        
        # Training samples
        self.training_samples_count.set(n_samples)
        
        # Hyperparameter trials
        if hyperparameter_trials > 0:
            for _ in range(hyperparameter_trials):
                self.hyperparameter_trials.labels(model_type=best_model_name).inc()
    
    def record_prediction_metrics(self, predictions, prediction_time, evaluation_metrics=None):
        """Record comprehensive prediction metrics"""
        if not self.monitoring_enabled:
            return
            
        # Prediction count and latency
        self.prediction_count.inc(len(predictions))
        latency_per_sample = prediction_time / len(predictions)
        
        for _ in range(len(predictions)):
            self.prediction_latency.observe(latency_per_sample)
        
        # RUL distribution
        for pred in predictions:
            self.rul_prediction_distribution.observe(pred)
        
        # Evaluation metrics
        if evaluation_metrics:
            for metric_name, value in evaluation_metrics.items():
                self.prediction_accuracy_metrics.labels(metric_type=metric_name).set(value)

# Enhanced decorator for stage monitoring
def monitor_pipeline_stage(stage_name):
    """Enhanced decorator for automatic stage monitoring"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get enhanced metrics instance
            metrics = None
            if hasattr(args[0], 'metrics'):
                metrics = args[0].metrics
            else:
                metrics = TurbineMLOpsMetrics()
            
            start_time = time.time()
            status = 'success'
            
            try:
                print(f"🚀 Starting stage: {stage_name}")
                
                # Update system metrics before stage execution
                if hasattr(metrics, 'update_system_metrics'):
                    metrics.update_system_metrics()
                
                result = func(*args, **kwargs)
                print(f"✅ Completed stage: {stage_name}")
                return result
            except Exception as e:
                status = 'failure'
                print(f"❌ Failed stage: {stage_name} - {e}")
                raise
            finally:
                duration = time.time() - start_time
                
                # Extract data shape if available from result
                data_shape = None
                if isinstance(result, tuple) and len(result) > 0:
                    # Try to get shape from first result if it's a dataframe path
                    pass  # Shape extraction logic can be added here
                
                if metrics:
                    metrics.record_stage_completion(stage_name, duration, status, data_shape)
                    metrics.push_metrics()
        
        return wrapper
    return decorator