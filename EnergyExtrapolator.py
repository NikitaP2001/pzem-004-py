import numpy as np

class EnergyExtrapolator:
    """
    Implements Low-Utilization Extrapolation with dynamic baseline discovery for accurate
    energy attribution based on CPU usage patterns.
    
    Uses a piecewise linear model to account for non-linear power scaling with utilization.
    """
    
    def __init__(self, window_size=10, min_utilization=0.5, baseline_percentile=5, 
                 high_util_threshold=80, baseline_window_size=25):
        """
        Initialize the piecewise linear energy extrapolation model with baseline discovery.
        
        Args:
            window_size: Size of moving average window for stabilizing readings (default: 10)
            min_utilization: Minimum total utilization to prevent division issues (default: 0.5)
            baseline_percentile: Percentile to use for baseline power estimation (default: 5)
            high_util_threshold: Threshold for high utilization regime in percentage (default: 80)
            baseline_window_size: Window size for baseline power detection (default: 25)
        """
        self.window_size = window_size
        self.min_utilization = min_utilization
        self.baseline_percentile = baseline_percentile
        self.high_util_threshold = high_util_threshold
        self.baseline_window_size = baseline_window_size
        
        # Moving window for CPU utilization data
        self.task_util_window = []
        self.total_util_window = []
        self.power_window = []
        
        # Statistical tracking
        self.utilization_ratios = []
        self.power_estimates = []
        
        # Baseline tracking
        self.dynamic_baseline = None
        self.power_at_low_util = []
        self.power_at_high_util = []
        self.utilization_segments = []
        
        # Model coefficients
        self.model_slope = None
        self.model_intercept = None
    
    def _update_baseline(self, total_power, total_util):
        """Update the dynamic baseline based on low utilization periods"""
        # Track power measurements during low utilization
        if total_util < 10.0:  # Consider very low utilization for baseline
            self.power_at_low_util.append(total_power)
            if len(self.power_at_low_util) > self.baseline_window_size:
                self.power_at_low_util.pop(0)
                
            # Update dynamic baseline when we have enough low-utilization samples
            if len(self.power_at_low_util) >= 5:
                # Use percentile rather than minimum to avoid outliers
                self.dynamic_baseline = np.percentile(self.power_at_low_util, self.baseline_percentile)
        
        # Track power at high utilization for model calibration
        if total_util > self.high_util_threshold:
            self.power_at_high_util.append((total_util, total_power))
            if len(self.power_at_high_util) > self.baseline_window_size:
                self.power_at_high_util.pop(0)
        
        # Add to utilization segments for distribution analysis
        self.utilization_segments.append(total_util)
        if len(self.utilization_segments) > self.baseline_window_size * 2:
            self.utilization_segments.pop(0)
    
    def _update_model_coefficients(self):
        """Update piecewise linear model coefficients based on observed data"""
        if not self.power_at_high_util or self.dynamic_baseline is None:
            return
            
        # Use the average high utilization point
        high_utils = [u for u, _ in self.power_at_high_util]
        high_powers = [p for _, p in self.power_at_high_util]
        
        if not high_utils or not high_powers:
            return
            
        avg_high_util = sum(high_utils) / len(high_utils)
        avg_high_power = sum(high_powers) / len(high_powers)
        
        # Assume linear relationship between baseline (at 0% util) and high util point
        # Power = slope * Utilization + baseline
        if avg_high_util > 0:
            self.model_slope = (avg_high_power - self.dynamic_baseline) / avg_high_util
            self.model_intercept = self.dynamic_baseline
    
    def calculate_task_power(self, total_power, task_util, total_util):
        """
        Calculate task power using a piecewise linear model with dynamic baseline discovery.
        
        Args:
            total_power: Total system power measurement in watts (float)
            task_util: Task CPU utilization percentage (float, 0-100)
            total_util: Total CPU utilization percentage (float, 0-100)
            
        Returns:
            float: Estimated task power consumption in watts
        """
        # Ensure values are primitive types, not pandas/numpy types
        total_power = float(total_power)
        task_util = float(task_util)
        total_util = float(total_util)
        
        # Ensure minimum utilization to avoid division issues
        total_util = max(self.min_utilization, total_util)
        
        # Ensure task_util doesn't exceed total_util
        task_util = min(task_util, total_util)
        
        # Update sliding windows
        self.task_util_window.append(task_util)
        self.total_util_window.append(total_util)
        self.power_window.append(total_power)
        
        # Keep window at specified size
        if len(self.task_util_window) > self.window_size:
            self.task_util_window.pop(0)
            self.total_util_window.pop(0)
            self.power_window.pop(0)
        
        # Update baseline power estimate
        self._update_baseline(total_power, total_util)
        
        # Update model coefficients
        self._update_model_coefficients()
        
        # Calculate moving averages to smooth data
        avg_task_util = sum(self.task_util_window) / len(self.task_util_window)
        avg_total_util = sum(self.total_util_window) / len(self.total_util_window)
        avg_total_power = sum(self.power_window) / len(self.power_window)
        
        # Calculate task power based on the current state of the model
        if self.dynamic_baseline is not None and self.model_slope is not None:
            # Piecewise linear model: 
            # 1. Subtract baseline power
            # 2. Attribute remaining power proportionally to utilization
            cpu_dependent_power = avg_total_power - self.dynamic_baseline
            cpu_dependent_power = max(0, cpu_dependent_power)  # Ensure non-negative
            
            # Calculate utilization ratio
            util_ratio = avg_task_util / avg_total_util if avg_total_util > 0 else 0
            self.utilization_ratios.append(util_ratio)
            
            # Task power = baseline portion + CPU-dependent portion
            # Where baseline portion = 0 (baseline power belongs to the system)
            task_power = cpu_dependent_power * util_ratio
            
            # Alternative model: directly model task power from task utilization
            # This can sometimes be more accurate if the relationship is well-defined
            model_based_task_power = self.model_slope * avg_task_util
            
            # Use the more conservative estimate between the two approaches
            task_power = min(task_power, model_based_task_power)
        else:
            # Fall back to simple ratio approach if we don't have enough data for the model
            util_ratio = avg_task_util / avg_total_util if avg_total_util > 0 else 0
            self.utilization_ratios.append(util_ratio)
            task_power = avg_total_power * util_ratio
        
        self.power_estimates.append(task_power)
        return task_power
    
    def process_data(self, total_power_values, task_util_values, total_util_values):
        """
        Process arrays of measurement data to calculate task power.
        
        Args:
            total_power_values: List/array of total power measurements (watts)
            task_util_values: List/array of task CPU utilization percentages (0-100)
            total_util_values: List/array of total CPU utilization percentages (0-100)
            
        Returns:
            list: Estimated task power values for each measurement point
        """
        # Reset state for new data processing
        self.task_util_window = []
        self.total_util_window = []
        self.power_window = []
        self.utilization_ratios = []
        self.power_estimates = []
        self.power_at_low_util = []
        self.power_at_high_util = []
        self.utilization_segments = []
        self.dynamic_baseline = None
        self.model_slope = None
        self.model_intercept = None
        
        # Pre-process data to establish baseline if there's enough data
        if len(total_power_values) > self.baseline_window_size:
            # First pass to establish baseline
            for i in range(len(total_power_values)):
                if total_util_values[i] < 10.0:
                    self.power_at_low_util.append(total_power_values[i])
                    
                if total_util_values[i] > self.high_util_threshold:
                    self.power_at_high_util.append((total_util_values[i], total_power_values[i]))
                    
                self.utilization_segments.append(total_util_values[i])
                    
            # Calculate initial baseline if we have enough low utilization samples
            if len(self.power_at_low_util) >= 3:
                self.dynamic_baseline = np.percentile(self.power_at_low_util, self.baseline_percentile)
                self._update_model_coefficients()
        
        # Process each data point
        task_power_values = []
        for i in range(len(total_power_values)):
            task_power = self.calculate_task_power(
                total_power_values[i], 
                task_util_values[i], 
                total_util_values[i]
            )
            task_power_values.append(task_power)
        
        return task_power_values
    
    def get_statistics(self):
        """
        Get statistical information about the extrapolation process.
        
        Returns:
            dict: Statistical measures of the extrapolation
        """
        if not self.utilization_ratios:
            return {
                "mean_util_ratio": None,
                "min_util_ratio": None,
                "max_util_ratio": None,
                "mean_task_power": None,
                "dynamic_baseline": None,
                "model_slope": None,
                "model_quality": "insufficient_data"
            }
            
        # Calculate utilization distribution to assess model quality
        util_distribution = {}
        if self.utilization_segments:
            low_util = len([u for u in self.utilization_segments if u < 30]) / len(self.utilization_segments)
            mid_util = len([u for u in self.utilization_segments if 30 <= u < 70]) / len(self.utilization_segments)
            high_util = len([u for u in self.utilization_segments if u >= 70]) / len(self.utilization_segments)
            util_distribution = {"low": low_util, "mid": mid_util, "high": high_util}
        
        # Assess model quality
        model_quality = "insufficient_data"
        if self.dynamic_baseline is not None:
            if len(self.power_at_low_util) >= 5 and len(self.power_at_high_util) >= 5:
                model_quality = "good"
            elif len(self.power_at_low_util) >= 3:
                model_quality = "fair"
            else:
                model_quality = "poor"
        
        return {
            "mean_util_ratio": sum(self.utilization_ratios) / len(self.utilization_ratios),
            "min_util_ratio": min(self.utilization_ratios),
            "max_util_ratio": max(self.utilization_ratios),
            "mean_task_power": sum(self.power_estimates) / len(self.power_estimates),
            "dynamic_baseline": self.dynamic_baseline,
            "model_slope": self.model_slope,
            "model_quality": model_quality,
            "utilization_distribution": util_distribution
        }