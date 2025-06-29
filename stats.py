import numpy as np
from scipy import stats
from scipy.stats import sem
from scipy.signal import find_peaks

def calculate_cv(data):
    """
    Calculate Coefficient of Variation (CV)
    CV = standard deviation / mean
    Lower values indicate more stable measurements
    """
    if data.mean() == 0:
        return float('inf')  # Avoid division by zero
    return data.std() / data.mean()

def calculate_sem(data):
    """
    Calculate Standard Error of the Mean (SEM)
    SEM = standard deviation / sqrt(n)
    Measures how precisely we know the true mean
    """
    return sem(data)

def calculate_snr(data):
    """
    Calculate Signal-to-Noise Ratio (SNR)
    Higher SNR indicates cleaner measurements
    """
    if data.std() == 0:
        return float('inf')  # Avoid division by zero
    return data.mean() / data.std()

def calculate_allan_deviation(data, tau=1):
    """
    Calculate Allan Deviation to evaluate stability across different time scales
    Useful for identifying drift effects
    
    Args:
        data: Time series data
        tau: Time lag (default=1)
    """
    # Ensure data is a numpy array
    y = np.array(data)
    
    # Calculate differences
    n = len(y)
    if n <= tau:
        return float('nan')
        
    # Create groups
    m = int(n / tau)
    groups = np.array([np.mean(y[i*tau:(i+1)*tau]) for i in range(m)])
    
    # Calculate Allan variance
    ad = np.sqrt(np.sum(np.diff(groups)**2) / (2 * (m-1)))
    return ad

def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate confidence interval for power measurements
    
    Args:
        data: Time series data of power measurements
        confidence: Confidence level (default=0.95 for 95% confidence)
    
    Returns:
        tuple: (lower_bound, upper_bound, margin_of_error)
    """
    n = len(data)
    if n <= 1:
        return (float('nan'), float('nan'), float('nan'))
    
    mean = data.mean()
    std_err = sem(data)
    
    # Get critical value from t-distribution
    t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin_of_error = t_crit * std_err
    
    return (mean - margin_of_error, mean + margin_of_error, margin_of_error)

def calculate_required_sample_size(data, total_duration_seconds, target_accuracy=0.5, confidence=0.95):
    """
    Calculate required sample size to achieve target accuracy with given confidence
    
    For a stationary power signal, if you need accuracy within ±ε watts 
    with 95% confidence, you need: n ≥ (1.96σ/ε)²
    
    Args:
        data: Time series data of power measurements
        total_duration_seconds: Actual duration of measurement in seconds
        target_accuracy: Target accuracy in watts or percentage (default=0.5%)
        confidence: Confidence level (default=0.95 for 95% confidence)
        
    Returns:
        dict: Sample size info and optimal measurement duration
    """
    # Check if data is empty
    if len(data) == 0:
        return {
            "required_samples": float('nan'),
            "current_samples": 0,
            "target_accuracy_watts": float('nan'),
            "current_accuracy_watts": float('nan'),
            "current_accuracy_pct": float('nan'),
            "sampling_freq": 0,
            "optimal_duration_seconds": float('nan')
        }
    
    # Calculate standard deviation
    std_dev = data.std()
    mean = data.mean()
    
    # Handle the case where std_dev is 0 or mean is 0
    if std_dev == 0 or mean == 0:
        return {
            "required_samples": 1,  # If std_dev is 0, we only need 1 sample
            "current_samples": len(data),
            "target_accuracy_watts": 0 if mean == 0 else target_accuracy,
            "current_accuracy_watts": 0,
            "current_accuracy_pct": 0,
            "sampling_freq": len(data) / total_duration_seconds if total_duration_seconds > 0 else 0,
            "optimal_duration_seconds": 0
        }
    
    # Convert target_accuracy from percentage to watts if it's under 1 (assuming it's a percentage)
    epsilon = target_accuracy if target_accuracy >= 1 else (target_accuracy/100) * mean
    
    # Get critical value based on confidence level (e.g., 1.96 for 95% confidence)
    z_crit = stats.norm.ppf((1 + confidence) / 2)
    
    # Calculate required sample size: n ≥ (z_crit * σ / ε)²
    # Handle potential NaN or Inf values
    try:
        required_n = int(np.ceil((z_crit * std_dev / epsilon) ** 2))
    except (ValueError, OverflowError):
        required_n = float('nan')
    
    # Calculate current accuracy with existing samples
    try:
        current_margin_of_error = z_crit * std_dev / np.sqrt(len(data))
        current_accuracy_watts = current_margin_of_error
        current_accuracy_pct = (current_margin_of_error / mean) * 100 if mean > 0 else float('nan')
    except (ValueError, ZeroDivisionError):
        current_margin_of_error = float('nan')
        current_accuracy_watts = float('nan')
        current_accuracy_pct = float('nan')
    
    # Sampling frequency estimation (samples per second)
    # Avoid division by zero
    sampling_freq = len(data) / total_duration_seconds if total_duration_seconds > 0 else 0
    
    # Calculate optimal duration
    optimal_duration = required_n / sampling_freq if sampling_freq > 0 else float('nan')
    
    return {
        "required_samples": required_n,
        "current_samples": len(data),
        "target_accuracy_watts": epsilon,
        "current_accuracy_watts": current_accuracy_watts,
        "current_accuracy_pct": current_accuracy_pct,
        "sampling_freq": sampling_freq,
        "optimal_duration_seconds": optimal_duration
    }

def evaluate_measurement_stability(data, hardware_accuracy=0.5):
    """
    Evaluate if the measurement stability meets or exceeds the hardware accuracy
    
    Args:
        data: Time series data of power measurements
        hardware_accuracy: Hardware accuracy specification (percentage)
        
    Returns:
        dict: Stability evaluation results
    """
    mean = data.mean()
    std_dev = data.std()
    cv = calculate_cv(data)
    
    # Calculate actual measurement uncertainty (statistical)
    actual_uncertainty_pct = (2 * std_dev / mean) * 100 if mean > 0 else float('nan')
    
    # PZEM-004 accuracy is specified as 0.5%
    hardware_uncertainty = hardware_accuracy
    
    # Determine if statistical uncertainty is better than hardware accuracy
    is_stable = actual_uncertainty_pct <= hardware_uncertainty
    
    # Calculate how many times the measurement is better/worse than hardware accuracy
    accuracy_ratio = hardware_uncertainty / actual_uncertainty_pct if actual_uncertainty_pct > 0 else float('inf')
    
    return {
        "mean_power": mean,
        "statistical_uncertainty_pct": actual_uncertainty_pct,
        "hardware_uncertainty_pct": hardware_uncertainty,
        "meets_hardware_accuracy": is_stable,
        "accuracy_ratio": accuracy_ratio,
        "coefficient_of_variation": cv
    }