# src/Turbine_RUL/monitoring/docker_metrics.py
# Simple monitoring that works in your Docker environment

import os
import time
from prometheus_client import Gauge, Counter, push_to_gateway, CollectorRegistry

class DockerMLOpsMetrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # Get Pushgateway URL from environment (set in docker-compose)
        self.pushgateway_url = os.getenv('PUSHGATEWAY_URL', 'localhost:9091')
        self.monitoring_enabled = os.getenv('PROMETHEUS_ENABLED', 'false').lower() == 'true'
        
        if self.monitoring_enabled:
            print(f"✅ Monitoring enabled - Pushgateway: {self.pushgateway_url}")
        else:
            print("⚠️ Monitoring disabled")
        
        # Define metrics
        self.stage_duration = Gauge(
            'pipeline_stage_duration_seconds',
            'Time taken for each pipeline stage',
            ['stage_name', 'status'],
            registry=self.registry
        )
        
        self.stage_counter = Counter(
            'pipeline_stage_executions_total',
            'Total executions of each stage',
            ['stage_name', 'status'],
            registry=self.registry
        )
        
        self.drift_detected = Gauge(
            'drift_detected',
            'Whether drift was detected (1=yes, 0=no)',
            ['drift_type'],
            registry=self.registry
        )
        
        self.data_quality = Gauge(
            'data_quality_score',
            'Data quality score (0-1)',
            ['dataset_type'],
            registry=self.registry
        )
        
        self.model_performance = Gauge(
            'model_rmse_score',
            'Model RMSE performance',
            ['split_type'],
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
            print(f"📊 Metrics pushed to {self.pushgateway_url}")
        except Exception as e:
            print(f"⚠️ Failed to push metrics: {e}")
    
    def record_stage_completion(self, stage_name, duration, status='success'):
        """Record stage completion metrics"""
        if not self.monitoring_enabled:
            return
            
        self.stage_duration.labels(stage_name=stage_name, status=status).set(duration)
        self.stage_counter.labels(stage_name=stage_name, status=status).inc()
        print(f"📈 Recorded {stage_name}: {duration:.2f}s ({status})")

# Decorator for easy stage monitoring
def monitor_pipeline_stage(stage_name):
    """Decorator to automatically monitor pipeline stages"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get metrics instance from self if available
            metrics = None
            if hasattr(args[0], 'metrics'):
                metrics = args[0].metrics
            else:
                metrics = DockerMLOpsMetrics()
            
            start_time = time.time()
            status = 'success'
            
            try:
                print(f"🚀 Starting stage: {stage_name}")
                result = func(*args, **kwargs)
                print(f"✅ Completed stage: {stage_name}")
                return result
            except Exception as e:
                status = 'failure'
                print(f"❌ Failed stage: {stage_name} - {e}")
                raise
            finally:
                duration = time.time() - start_time
                if metrics:
                    metrics.record_stage_completion(stage_name, duration, status)
                    metrics.push_metrics()
        
        return wrapper
    return decorator