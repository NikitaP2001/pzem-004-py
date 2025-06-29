def print_summary(summary):
    print("\n=== Measurement Summary ===")
    print(f"Task Power: {summary.get('taskPower', 'N/A'):.2f} W")
    print(f"Total Power: {summary.get('totalPower', 'N/A'):.2f} W")
    print(f"Task Energy: {summary.get('taskEnergy', 'N/A'):.2f} J")
    print(f"Total Energy: {summary.get('totalEnergy', 'N/A'):.2f} J")
    print(f"Measurement Duration: {summary.get('measurement_duration', 'N/A'):.2f} seconds")
    
    print("\n=== Power Stability Metrics ===")
    print(f"Coefficient of Variation: {summary.get('totalPower_cv', 'N/A'):.6f}")
    print(f"Standard Error of Mean: {summary.get('totalPower_sem', 'N/A'):.6f}")
    print(f"Signal-to-Noise Ratio: {summary.get('totalPower_snr', 'N/A'):.2f}")
    print(f"Allan Deviation: {summary.get('totalPower_allan', 'N/A'):.6f}")
    
    print("\n=== Power Accuracy Analysis ===")
    print(f"Mean Power: {summary.get('totalPower', 'N/A'):.2f} W")
    print(f"95% Confidence Interval: [{summary.get('totalPower_ci_lower', 'N/A'):.2f}, "
          f"{summary.get('totalPower_ci_upper', 'N/A'):.2f}] W")
    print(f"Margin of Error: ±{summary.get('totalPower_margin_of_error', 'N/A'):.4f} W")
    
    print("\n=== Sampling Requirements ===")
    print(f"Current Samples: {summary.get('totalPower_current_samples', 'N/A')}")
    print(f"Required Samples for 0.5% Accuracy: {summary.get('totalPower_required_samples', 'N/A')}")
    print(f"Current Accuracy: ±{summary.get('totalPower_current_accuracy_watts', 'N/A'):.4f} W "
          f"({summary.get('totalPower_current_accuracy_pct', 'N/A'):.2f}%)")
    print(f"Actual Sampling Frequency: {summary.get('totalPower_sampling_freq', 'N/A'):.2f} Hz")
    print(f"Optimal Measurement Duration: {summary.get('totalPower_optimal_duration_seconds', 'N/A'):.1f} seconds")
    
    print("\n=== Hardware Accuracy Comparison ===")
    print(f"Statistical Uncertainty: ±{summary.get('totalPower_statistical_uncertainty_pct', 'N/A'):.2f}%")
    print(f"Hardware Uncertainty: ±{summary.get('totalPower_hardware_uncertainty_pct', 'N/A'):.2f}%")
    print(f"Meets Hardware Accuracy: {summary.get('totalPower_meets_hardware_accuracy', 'N/A')}")
    print(f"Accuracy Ratio (Hardware/Statistical): {summary.get('totalPower_accuracy_ratio', 'N/A'):.2f}x")
    
    print("\n=== CPU Metrics ===")
    print(f"Mean CPU: {summary.get('totalCPU_mean', 'N/A'):.2f}%")
    print(f"CPU Coefficient of Variation: {summary.get('totalCPU_cv', 'N/A'):.6f}")
    print(f"CPU Standard Error of Mean: {summary.get('totalCPU_sem', 'N/A'):.6f}")
    print(f"CPU Signal-to-Noise Ratio: {summary.get('totalCPU_snr', 'N/A'):.2f}")