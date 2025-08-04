import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def calculate_RUL(X, upper_threshold=None):
    """Calculate Remaining Useful Life per unit - Google Colab version"""
    lifetime = X.groupby(['unit_id'])['time_cycles'].transform(max)
    rul = lifetime - X['time_cycles']
    
    if upper_threshold:
        rul = np.where(rul > upper_threshold, upper_threshold, rul)
    
    return rul

def calculate_evaluation_metrics(true_rul_series, pred_rul, engine_ids):
    """Calculate evaluation metrics for RUL predictions"""
    # Align predictions with true RUL values
    true_rul_aligned = []
    pred_rul_aligned = []
    
    for i, engine_id in enumerate(engine_ids):
        if engine_id in true_rul_series.index:
            true_rul_aligned.append(true_rul_series[engine_id])
            pred_rul_aligned.append(pred_rul[i])
    
    if len(true_rul_aligned) == 0:
        print("⚠️ No matching engines for evaluation")
        return None, None, None
    
    true_rul_array = np.array(true_rul_aligned)
    pred_rul_array = np.array(pred_rul_aligned)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(true_rul_array, pred_rul_array))
    mae = mean_absolute_error(true_rul_array, pred_rul_array)
    
    # MAPE with zero handling
    non_zero_mask = true_rul_array != 0
    mape = mean_absolute_percentage_error(true_rul_array[non_zero_mask], 
                                        pred_rul_array[non_zero_mask]) if np.sum(non_zero_mask) > 0 else float('inf')
    
    # Custom RUL score
    def rul_score_f(err):
        return np.exp(err / 10) - 1 if err >= 0 else np.exp(-err / 13) - 1
    
    err = pred_rul_array - true_rul_array
    custom_score = np.sum([rul_score_f(x) for x in err])
    
    metrics = {
        'rmse': rmse, 'mae': mae, 'mape': mape,
        'accuracy': (1 - mape) * 100 if mape != float('inf') else 0,
        'custom_score': custom_score, 'n_engines': len(true_rul_array),
        'pred_range': (pred_rul_array.min(), pred_rul_array.max()),
        'true_range': (true_rul_array.min(), true_rul_array.max())
    }
    
    return metrics, true_rul_array, pred_rul_array

def create_evaluation_plots(true_rul, pred_rul, metrics, plots_save_path):
    """Create evaluation plots for RUL predictions"""
    print("Creating evaluation plots...")
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        residuals = pred_rul - true_rul
        
        # Plot 1: Predictions vs True
        axes[0,0].scatter(true_rul, pred_rul, alpha=0.6, s=30)
        min_val, max_val = min(true_rul.min(), pred_rul.min()), max(true_rul.max(), pred_rul.max())
        axes[0,0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[0,0].set_xlabel('True RUL')
        axes[0,0].set_ylabel('Predicted RUL')
        axes[0,0].set_title('Predicted vs True RUL')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Residuals
        axes[0,1].scatter(true_rul, residuals, alpha=0.6, s=30)
        axes[0,1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0,1].set_xlabel('True RUL')
        axes[0,1].set_ylabel('Residuals')
        axes[0,1].set_title('Residuals vs True RUL')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Error distribution
        axes[0,2].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[0,2].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0,2].axvline(x=residuals.mean(), color='orange', linestyle='-', linewidth=2)
        axes[0,2].set_xlabel('Prediction Error')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].set_title('Error Distribution')
        axes[0,2].grid(True, alpha=0.3)
        
        # Plot 4: Absolute Error vs True RUL
        axes[1,0].scatter(true_rul, np.abs(residuals), alpha=0.6, s=30)
        axes[1,0].set_xlabel('True RUL')
        axes[1,0].set_ylabel('Absolute Error')
        axes[1,0].set_title('Absolute Error vs True RUL')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 5: Error by RUL range
        bins = np.linspace(true_rul.min(), true_rul.max(), 10)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_errors = []
        
        for i in range(len(bins)-1):
            mask = (true_rul >= bins[i]) & (true_rul < bins[i+1])
            bin_errors.append(np.abs(residuals[mask]).mean() if np.sum(mask) > 0 else 0)
        
        axes[1,1].bar(bin_centers, bin_errors, width=(bins[1]-bins[0])*0.8, alpha=0.7)
        axes[1,1].set_xlabel('True RUL Range')
        axes[1,1].set_ylabel('Mean Absolute Error')
        axes[1,1].set_title('Error by RUL Range')
        axes[1,1].grid(True, alpha=0.3)
        
        # Plot 6: Metrics summary
        axes[1,2].axis('off')
        metrics_text = f"""EVALUATION METRICS
RMSE: {metrics['rmse']:.2f}
MAE: {metrics['mae']:.2f}
MAPE: {metrics['mape']:.3f}
Accuracy: {metrics['accuracy']:.1f}%
Custom Score: {metrics['custom_score']:.2f}
Engines: {metrics['n_engines']}"""
        
        axes[1,2].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(plots_save_path), exist_ok=True)
        plt.savefig(plots_save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Plots saved: {plots_save_path}")
        
    except Exception as e:
        print(f"⚠️ Plotting failed: {e}")

def save_prediction_results(predictions, engine_ids, true_rul_series, metrics, 
                          predictions_path, metrics_path):
    """Save prediction results and evaluation metrics"""
    print("Saving results...")
    
    results_data = {'engine_id': engine_ids, 'predicted_rul': predictions}
    
    if true_rul_series is not None:
        true_vals, errors, abs_errors, perc_errors = [], [], [], []
        
        for engine_id, pred in zip(engine_ids, predictions):
            if engine_id in true_rul_series.index:
                true_val = true_rul_series[engine_id]
                error = pred - true_val
                abs_error = abs(error)
                perc_error = abs_error / true_val * 100 if true_val != 0 else float('inf')
            else:
                true_val = error = abs_error = perc_error = np.nan
            
            true_vals.append(true_val)
            errors.append(error)
            abs_errors.append(abs_error)
            perc_errors.append(perc_error)
        
        results_data.update({
            'true_rul': true_vals, 'error': errors,
            'abs_error': abs_errors, 'percentage_error': perc_errors
        })
    
    results_df = pd.DataFrame(results_data)
    
    # Save files
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
    results_df.to_csv(predictions_path, index=False)
    
    if metrics is not None:
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    
    print(f"✅ Results saved: {predictions_path}")
    return results_df

def extract_and_save_true_rul(raw_test_data, evaluation_rul_path):
    """Extract true RUL values before preprocessing and save them"""
    if 'rul' not in raw_test_data.columns:
        print("⚠️ No RUL column found - evaluation will be skipped")
        return None
        
    true_rul_per_engine = raw_test_data.groupby('unit_id')['rul'].last()
    os.makedirs(os.path.dirname(evaluation_rul_path), exist_ok=True)
    true_rul_per_engine.to_csv(evaluation_rul_path)
    print(f"✅ True RUL values saved: {len(true_rul_per_engine)} engines")
    return true_rul_per_engine

def print_evaluation_summary(metrics):
    """Print evaluation summary"""
    print(f"\n{'='*60}")
    print("🎯 MODEL EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Engines: {metrics['n_engines']} | RMSE: {metrics['rmse']:.2f} | MAE: {metrics['mae']:.2f}")
    if metrics['mape'] != float('inf'):
        print(f"MAPE: {metrics['mape']:.3f} | Accuracy: {metrics['accuracy']:.1f}%")
    print(f"Custom Score: {metrics['custom_score']:.2f}")
    print(f"{'='*60}")