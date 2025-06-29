import pandas as pd
import os
import json
import glob
from print import print_summary
from MeasurementRunner import MeasurementRunner
from MeasurementPlotter import MeasurementPlotter
from MeasurementStatistics import MeasurementStatistics
    
import argparse

def parse_arguments():
    """
    Parse and validate command-line arguments.
    Supported arguments:
    - -P <port_name>: The name of the serial port (default: /dev/ttyACM0).
    - -T <seconds>: Time limit for a measurement to run (integer, mutually exclusive with -E).
    - -E <exec string>: Command to execute while the program runs (string, mutually exclusive with -T).
    Returns:
    - dict: A dictionary containing parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="A program for measuring and plotting data from a serial port."
    )

    # Define arguments
    parser.add_argument(
        "-P", "--port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port name (default: /dev/ttyACM0)."
    )
    parser.add_argument(
        "-T", "--time",
        type=int,
        help="Time limit for the measurement in seconds (mutually exclusive with -E)."
    )
    parser.add_argument(
        "-E", "--exec",
        type=str,
        help="Command to execute while the program runs (mutually exclusive with -T)."
    )
    parser.add_argument(
        "-H", "--hep",
        action="store_true",
        help="Hep score load measurement"
    )
    parser.add_argument(
        "-F", "--floor",
        action="store_true",
        help="Account baseline power in measurement"
    )

    parser.add_argument(
        "-G", "--gui",
        action="store_true",
        help="Show GUI plot report"
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate mutually exclusive arguments
    if args.time is not None and args.exec is not None:
        parser.error("Arguments -T and -E are mutually exclusive. Specify only one.")

    if args.hep:
        args.exec="sudo hep-score -v -m docker -f hepscore_short.yaml ./testdir"

    # Organize arguments into a dictionary
    params = {
        "port": args.port,
        "time": args.time,
        "exec": args.exec,
        "floor": args.floor,
        "hep": args.hep,
        "plot": args.gui
    }

    return params

import pandas as pd

def power_get_floor(stat: MeasurementStatistics):
    totalPower = stat.get('totalPower')
    floor_power = totalPower.mean()
    return floor_power

def measure_floor_power(target_accuracy=0.5, max_duration=100, start_duration=10) -> float:

    print("\n=== Adaptive Floor Power Measurement ===")
    print(f"Target accuracy: {target_accuracy}% (hardware spec)")
    
    # Try progressively longer durations
    durations = [start_duration]
    current_duration = start_duration
    while current_duration < max_duration:
        current_duration += 10
        durations.append(current_duration)
    
    best_measurement = None
    best_accuracy = float('inf')
    best_power = 0
    
    # Try each duration up to 5 times
    for duration in durations:
        print(f"\nTrying {duration} second measurements...")
        
        for attempt in range(1, 6):  # 5 attempts at each duration
            print(f"  Attempt {attempt}/5 with {duration}s duration")
            
            # Run the measurement with current duration
            params = {
                "port": "/dev/ttyACM0",
                "time": duration,
                "exec": None,
                "plot": False,
            }
            stat = MeasurementStatistics(params)
            reader = MeasurementRunner(params, stat)
            stat.start()
            reader.run()
            stat.end()
            
            # Filter out initial peak
            peak_duration = stat.filter_peaks()
            if peak_duration > 0:
                print(f"  Filtered initial peak of {peak_duration:.2f} seconds")
            
            # Get measurement summary and accuracy stats
            summary = stat.getSummary()
            actual_duration = summary.get('measurement_duration', 0)
            current_accuracy = summary.get('totalPower_current_accuracy_pct')
            statistical_uncertainty = summary.get('totalPower_statistical_uncertainty_pct')
            cv = summary.get('totalPower_cv')
            snr = summary.get('totalPower_snr')
            power_value = summary.get('totalPower')
            
            # Print key metrics for this measurement
            print(f"  Power: {power_value:.2f}W, Accuracy: ±{current_accuracy:.2f}%, SNR: {snr:.2f}")
            
            # Check if this is the best measurement so far
            if current_accuracy is not None and current_accuracy < best_accuracy:
                best_accuracy = current_accuracy
                best_measurement = stat
                best_power = power_value
                
                # If we've reached target accuracy, we can stop
                if current_accuracy <= target_accuracy:
                    print(f"\n✓ Target accuracy achieved! ({best_accuracy:.2f}% ≤ {target_accuracy}%)")
                    print(f"Final floor power value: {best_power:.2f}W")
                    return best_power
        
        # If we've tried all durations and none met the target, use the best one
        if duration == durations[-1]:
            break
    
    print(f"\nℹ️ Target accuracy not achieved. Using best measurement (±{best_accuracy:.2f}%)")
    print(f"Final floor power value: {best_power:.2f}W")
    
    # Print recommendation for better accuracy
    if best_accuracy > target_accuracy:
        optimal_duration = summary.get('totalPower_optimal_duration_seconds', 0)
        print(f"ℹ️ For {target_accuracy}% accuracy, approximately {int(optimal_duration)}s would be needed")
    
    return best_power

def measure_common(stat: MeasurementStatistics, params: dict):
    reader = MeasurementRunner(params, stat)
    stat.start()
    reader.run()
    stat.end()

def measure_single_round(stat: MeasurementStatistics, params: dict) -> dict:
    # Find the most recent HEPscore directory
    summary = ""
    reader = MeasurementRunner(params, stat)
    stat.start()
    reader.run()
    stat.end()

    hepscore_dirs = glob.glob("testdir/HEPscore_*")
    if hepscore_dirs:
        latest_dir = max(hepscore_dirs, key=os.path.getctime)
        json_file = os.path.join(latest_dir, "HEPscoreTestKV.json")
        
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r') as f:
                    hep_data = json.load(f)
                    
                energy_result = hep_data.get("energy")
                if energy_result is not None:
                    print(f"HEPscore energy result: {energy_result} J")
                    summary = stat.getSummary()
                    print(f"Measured energy: {summary['taskEnergy']} J")
                    print(f"Extr task energy: {summary['taskEnergyExtr']} J")
                    
                    if summary['taskEnergy'] is not None and energy_result > 0:
                        ratio = summary['taskEnergy'] / energy_result
                        print(f"Ratio (measured/reported): {ratio:.4f}")
            except Exception as e:
                print(f"Error processing HEPscore results: {e}")
        else:
            print(f"HEPscore results file not found: {json_file}")
    else:
        print("No HEPscore results directory found")
        summary = stat.getSummary()

    
    task_energy = int(summary['taskEnergy'])
    return { "pzemEnergy": task_energy, "hepEnergy": energy_result }

def measure_hepscore(stat: MeasurementStatistics, params: dict):
    sample_count = 1
    energy_df = pd.DataFrame(columns=["pzemEnergy", "hepEnergy"])
    for i in range(sample_count):
        row = measure_single_round(stat, params)
        energy_df = pd.concat([energy_df, pd.DataFrame([row])], ignore_index=True)
    
    os.makedirs("pzem_res", exist_ok=True)
    energy_df.to_csv("pzem_res/atlas-gen-bmk.csv", index=False)

def main():
    params = parse_arguments()
    floor_power:float = 0
    is_hepscore = params['hep']
    is_floor = params['floor']

    if is_floor:
        floor_power = measure_floor_power()

    stat = MeasurementStatistics(params, floor_power)

    if is_hepscore:
        measure_hepscore(stat, params)
    else:
        measure_common(stat, params)

    peak_dur = stat.filter_peaks()
    print(f"Filtered out initial peak of {peak_dur:.2f} seconds")

    summary = stat.getSummary()
    print_summary(summary)
    if params['plot']:
        plotter = MeasurementPlotter(stat)
        plotter.draw()

if __name__ == "__main__":
    main()