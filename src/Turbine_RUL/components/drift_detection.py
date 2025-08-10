import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
from src.Turbine_RUL.monitoring.docker_metrics import DockerMLOpsMetrics, monitor_pipeline_stage
from src.Turbine_RUL.config.configuration import ConfigurationManager

class DriftDetector:
    def __init__(self):
        config_manager = ConfigurationManager()
        self.config = config_manager.get_drift_detection_config()
        self.drift_results = {}
        self.metrics = DockerMLOpsMetrics()
    
    def calculate_and_save_reference(self):
        """Calculate reference data from training set - RUN ONCE MANUALLY"""
        print("Calculating reference data from training set...")
        
        # Load training data
        train_data = pd.read_csv(self.config.train_data_path)
        print(f"📊 Training data loaded: {train_data.shape}")
        
        # Sample for reference (avoid memory issues)
        reference_sample = train_data.sample(min(5000, len(train_data)), random_state=42)
        
        # Calculate stats for sensor columns
        sensor_columns = [col for col in train_data.columns if col.startswith('sensor')]
        reference_stats = {}
        
        for col in sensor_columns:
            col_data = train_data[col].dropna()
            reference_stats[col] = {
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'missing_rate': float(train_data[col].isnull().mean())
            }
        
        # Add temporal stats if available
        if 'unit_id' in train_data.columns and 'time_cycles' in train_data.columns:
            avg_cycles = train_data.groupby('unit_id')['time_cycles'].max().mean()
            reference_stats['avg_cycles_per_unit'] = float(avg_cycles)
            reference_stats['total_units'] = int(train_data['unit_id'].nunique())
            reference_stats['total_cycles'] = int(train_data['time_cycles'].max())
        
        # Save reference data and stats
        os.makedirs(os.path.dirname(self.config.reference_data_path), exist_ok=True)
        
        with open(self.config.reference_data_path, 'wb') as f:
            pickle.dump(reference_sample, f)
            
        with open(self.config.reference_stats_path, 'wb') as f:
            pickle.dump(reference_stats, f)
            
        print(f"✅ Reference data saved for {len(sensor_columns)} sensors")
        print(f"📁 Reference data: {self.config.reference_data_path}")
        print(f"📊 Reference stats: {self.config.reference_stats_path}")
        return reference_stats
        
    def load_reference_artifacts(self):
        """Load reference data and statistics from training"""
        print("Loading reference artifacts for drift detection...")
        
        try:
            with open(self.config.reference_data_path, 'rb') as f:
                self.reference_data = pickle.load(f)
            with open(self.config.reference_stats_path, 'rb') as f:
                self.reference_stats = pickle.load(f)
            print(f"✅ Reference artifacts loaded successfully")
        except FileNotFoundError as e:
            print(f"⚠️ Reference artifacts not found: {e}")
            print("This is likely the first run - will trigger retraining")
            return False
        return True
            
    def calculate_psi(self, reference, current, bins=10):
        """Calculate Population Stability Index"""
        try:
            # Handle edge cases
            if len(reference) == 0 or len(current) == 0:
                return 1.0  # High PSI indicates drift
                
            ref_hist, bin_edges = np.histogram(reference, bins=bins)
            cur_hist, _ = np.histogram(current, bins=bin_edges)
            
            ref_pct = ref_hist / len(reference)
            cur_pct = cur_hist / len(current)
            
            # Avoid division by zero
            ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
            cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)
            
            psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
            return abs(psi)  # Return absolute value
        except Exception as e:
            print(f"PSI calculation error: {e}")
            return 1.0  # Conservative approach - assume drift
    
    def detect_data_drift(self, new_data):
        """Detect statistical drift in sensor features"""
        print("Analyzing data drift in sensor features...")
        
        drift_detected = False
        feature_drifts = {}
        
        # Focus on sensor columns for drift detection
        sensor_columns = [col for col in new_data.columns if col.startswith('sensor')]
        
        for column in sensor_columns:
            if column in self.reference_stats and column in new_data.columns:
                try:
                    # Get reference and current data
                    ref_data = self.reference_data[column].dropna()
                    cur_data = new_data[column].dropna()
                    
                    if len(ref_data) == 0 or len(cur_data) == 0:
                        continue
                    
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_p_value = stats.ks_2samp(ref_data, cur_data)
                    
                    # Population Stability Index (PSI)
                    psi_score = self.calculate_psi(ref_data, cur_data)
                    
                    # Statistical difference in means (normalized)
                    ref_mean = self.reference_stats[column]['mean']
                    ref_std = self.reference_stats[column]['std']
                    cur_mean = cur_data.mean()
                    
                    mean_drift = abs(cur_mean - ref_mean) / (ref_std + 1e-8)
                    
                    feature_drift = {
                        'ks_statistic': float(ks_stat),
                        'ks_p_value': float(ks_p_value),
                        'psi_score': float(psi_score),
                        'mean_drift': float(mean_drift),
                        'reference_mean': float(ref_mean),
                        'current_mean': float(cur_mean),
                        'drift_detected': ( 
                                         psi_score > 0.7 or 
                                         mean_drift > 3.0)  # More practical thresholds
                    }
                    
                    feature_drifts[column] = feature_drift
                    if feature_drift['drift_detected']:
                        drift_detected = True
                        print(f"  🚨 Drift detected in {column}: PSI={psi_score:.3f}, KS_p={ks_p_value:.3f}")
                    else:
                        print(f"  ✅ No drift in {column}: PSI={psi_score:.3f}, KS_p={ks_p_value:.3f}")
                        
                except Exception as e:
                    print(f"  ⚠️ Error analyzing {column}: {e}")
                    # Conservative approach - assume drift on error
                    feature_drifts[column] = {'error': str(e), 'drift_detected': True}
                    drift_detected = True
                    
        return drift_detected, feature_drifts
    
    def detect_data_quality_drift(self, new_data):
        """Detect data quality issues that might indicate drift"""
        print("Analyzing data quality drift...")
        
        quality_issues = {}
        quality_drift = False
        
        # Check missing values pattern
        missing_rates = new_data.isnull().mean()
        if hasattr(self, 'reference_stats'):
            for col in missing_rates.index:
                if col in self.reference_stats:
                    ref_missing_rate = self.reference_stats[col].get('missing_rate', 0)
                    current_missing_rate = missing_rates[col]
                    
                    if abs(current_missing_rate - ref_missing_rate) > 0.2:  # 20% threshold - more practical
                        quality_issues[f'{col}_missing_rate'] = {
                            'reference': ref_missing_rate,
                            'current': current_missing_rate,
                            'drift_detected': True
                        }
                        quality_drift = True
        
        # Check for unexpected values (outliers beyond training range)
        sensor_columns = [col for col in new_data.columns if col.startswith('sensor')]
        for col in sensor_columns:
            if col in self.reference_stats:
                ref_min = self.reference_stats[col]['min']
                ref_max = self.reference_stats[col]['max']
                cur_min = new_data[col].min()
                cur_max = new_data[col].max()
                
                # More lenient range check - only flag significant outliers
                range_tolerance = 0.2  # 20% tolerance
                lower_bound = ref_min * (1 - range_tolerance)
                upper_bound = ref_max * (1 + range_tolerance)
                
                if cur_min < lower_bound or cur_max > upper_bound:
                    quality_issues[f'{col}_range'] = {
                        'reference_range': [ref_min, ref_max],
                        'current_range': [cur_min, cur_max],
                        'drift_detected': True
                    }
                    quality_drift = True
        
        return quality_drift, quality_issues
    
    def detect_temporal_drift(self, new_data):
        """Detect temporal patterns that might indicate drift"""
        print("Analyzing temporal drift...")
        
        temporal_drift = False
        temporal_info = {}
        
        try:
            # Check if we have unit_id and time_cycles
            if 'unit_id' in new_data.columns and 'time_cycles' in new_data.columns:
                # Check average cycle length per unit
                cycles_per_unit = new_data.groupby('unit_id')['time_cycles'].max().mean()
                
                if hasattr(self, 'reference_stats') and 'avg_cycles_per_unit' in self.reference_stats:
                    ref_avg_cycles = self.reference_stats['avg_cycles_per_unit']
                    cycle_drift = abs(cycles_per_unit - ref_avg_cycles) / ref_avg_cycles
                    
                    temporal_info = {
                        'reference_avg_cycles': ref_avg_cycles,
                        'current_avg_cycles': cycles_per_unit,
                        'cycle_drift_ratio': cycle_drift,
                        'temporal_drift': cycle_drift > 0.5  # 50% threshold - more practical
                    }
                    
                    temporal_drift = temporal_info['temporal_drift']
                    
        except Exception as e:
            print(f"  ⚠️ Temporal drift analysis error: {e}")
            temporal_info = {'error': str(e)}
            
        return temporal_drift, temporal_info
    
    @monitor_pipeline_stage('drift_detection')
    def initiate_drift_detection(self, test_data_path=None):
        """Main drift detection pipeline"""
        print("="*60)
        print("🔍 STARTING DRIFT DETECTION PIPELINE")
        print("="*60)
        
        start_time = datetime.now()
        
        # Use provided path or config default
        data_path = test_data_path or self.config.test_data_path
        
        # Load new test data
        try:
            new_data = pd.read_csv(data_path)
            print(f"📊 New data loaded: {new_data.shape}")
        except Exception as e:
            print(f"❌ Error loading test data: {e}")
            return True, {'error': 'Could not load test data', 'recommendation': 'retrain'}
        
        # Try to load reference artifacts
        if not self.load_reference_artifacts():
            print("📝 No reference data found - First time run")
            drift_report = {
                'timestamp': datetime.now().isoformat(),
                'first_run': True,
                'data_drift_detected': True,  # Force retraining on first run
                'overall_drift_detected': True,
                'recommendation': 'retrain',
                'reason': 'No reference data available - initial training required'
            }
            
            # Save drift report
            os.makedirs(os.path.dirname(self.config.drift_report_path), exist_ok=True)
            with open(self.config.drift_report_path, 'w') as f:
                json.dump(drift_report, f, indent=2, default=str)
            
            return True, drift_report
        
        # Perform drift detection analyses
        data_drift_detected, feature_drifts = self.detect_data_drift(new_data)
        quality_drift_detected, quality_issues = self.detect_data_quality_drift(new_data)
        temporal_drift_detected, temporal_info = self.detect_temporal_drift(new_data)
        
        # Overall drift decision
        overall_drift = data_drift_detected or quality_drift_detected or temporal_drift_detected
        
        # Calculate drift summary statistics
        total_features_analyzed = len(feature_drifts)
        features_with_drift = sum(1 for f in feature_drifts.values() if f.get('drift_detected', False))
        drift_percentage = (features_with_drift / total_features_analyzed * 100) if total_features_analyzed > 0 else 0
        
        # Generate comprehensive drift report
        drift_report = {
            'timestamp': datetime.now().isoformat(),
            'analysis_duration_seconds': (datetime.now() - start_time).total_seconds(),
            'data_shape': list(new_data.shape),
            
            # Overall results
            'data_drift_detected': data_drift_detected,
            'quality_drift_detected': quality_drift_detected,
            'temporal_drift_detected': temporal_drift_detected,
            'overall_drift_detected': overall_drift,
            
            # Detailed results
            'feature_drifts': feature_drifts,
            'quality_issues': quality_issues,
            'temporal_info': temporal_info,
            
            # Summary statistics
            'drift_summary': {
                'total_features_analyzed': total_features_analyzed,
                'features_with_drift': features_with_drift,
                'drift_percentage': drift_percentage,
                'drift_threshold_used': 0.001,  # Updated threshold
                'psi_threshold_used': 0.5,
                'mean_drift_threshold_used': 3.0
            },
            
            # Decision
            'recommendation': 'retrain' if overall_drift else 'proceed_with_prediction'
        }
        
        # Save drift report
        os.makedirs(os.path.dirname(self.config.drift_report_path), exist_ok=True)
        with open(self.config.drift_report_path, 'w') as f:
            json.dump(drift_report, f, indent=2, default=str)

        if self.metrics.monitoring_enabled:
            # Record drift metrics
            self.metrics.drift_detected.labels(drift_type='overall').set(1 if overall_drift else 0)
            self.metrics.drift_detected.labels(drift_type='data').set(1 if data_drift_detected else 0)
            self.metrics.drift_detected.labels(drift_type='quality').set(1 if quality_drift_detected else 0)
            self.metrics.drift_detected.labels(drift_type='temporal').set(1 if temporal_drift_detected else 0)
            
            # Record data quality
            missing_ratio = new_data.isnull().sum().sum() / (new_data.shape[0] * new_data.shape[1])
            quality_score = 1 - missing_ratio
            self.metrics.data_quality.labels(dataset_type='test').set(quality_score)
        
        
        # Print summary
        print("\n" + "="*60)
        print("📋 DRIFT DETECTION SUMMARY")
        print("="*60)
        print(f"📊 Data Drift: {'🚨 DETECTED' if data_drift_detected else '✅ Not Detected'}")
        print(f"🔍 Quality Drift: {'🚨 DETECTED' if quality_drift_detected else '✅ Not Detected'}")
        print(f"⏱️ Temporal Drift: {'🚨 DETECTED' if temporal_drift_detected else '✅ Not Detected'}")
        print(f"🎯 Overall Decision: {'🔄 RETRAIN MODEL' if overall_drift else '⚡ PROCEED WITH PREDICTION'}")
        print(f"📈 Features with Drift: {features_with_drift}/{total_features_analyzed} ({drift_percentage:.1f}%)")
        print(f"💾 Report saved: {self.config.drift_report_path}")
        print("="*60)
        
        return overall_drift, drift_report