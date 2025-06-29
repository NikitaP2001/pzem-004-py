#!/usr/bin/env python3
# filepath: /home/user/src/pzem-004-py/capture_stats.py

import subprocess
import os
import time
from datetime import datetime

def run_measurement(duration_seconds):
    """Run a single measurement with the specified duration"""
    print(f"\n=== Running measurement for {duration_seconds} seconds ===")
    
    # Command to run with sudo
    command = f"sudo -E python pzem_004t.py -T {duration_seconds}"
    
    # Generate filename with timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{duration_seconds}_sec_stats_{timestamp}.txt"
    
    print(f"Running command: {command}")
    print(f"Saving output to: {output_file}")
    
    try:
        # Execute the command and capture output
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True
        )
        
        # Save the output to file
        with open(output_file, "w") as f:
            f.write(f"# Measurement with duration: {duration_seconds} seconds\n")
            f.write(f"# Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Command: {command}\n\n")
            f.write(result.stdout)
            
        return True, output_file
    
    except subprocess.CalledProcessError as e:
        print(f"Error running measurement: {e}")
        # Save error output to file
        with open(output_file, "w") as f:
            f.write(f"# ERROR in measurement with duration: {duration_seconds} seconds\n")
            f.write(f"# Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Command: {command}\n\n")
            f.write(f"Error: {e}\n\n")
            if e.stdout:
                f.write(e.stdout)
        return False, output_file

def run_duration_set(duration_seconds, repetitions=5, cool_down=5):
    """Run multiple measurements of the same duration"""
    output_files = []
    
    print(f"\n=== Starting set of {repetitions} measurements for {duration_seconds} second duration ===")
    
    for i in range(1, repetitions+1):
        print(f"\nMeasurement {i}/{repetitions} for {duration_seconds} seconds:")
        success, output_file = run_measurement(duration_seconds)
        output_files.append(output_file)
        
        # Cool down between measurements
        if i < repetitions:
            print(f"Cooling down for {cool_down} seconds before next measurement...")
            time.sleep(cool_down)
    
    # Create a summary file for this duration
    summary_file = f"{duration_seconds}_sec_summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"# Summary for {duration_seconds} second measurements\n")
        f.write(f"# Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Files: {', '.join(output_files)}\n\n")
        
        # Additional analysis could be added here
        
    return output_files

def main():
    """Main function to run all measurement sets"""
    durations = [10, 20, 30, 40, 50]
    repetitions = 5
    cool_down_between_measurements = 5  # seconds
    cool_down_between_sets = 15  # seconds
    
    print("=== PC Power Statistics Collection ===")
    print(f"Will collect {repetitions} measurements for each duration: {durations} seconds")
    
    all_files = []
    for i, duration in enumerate(durations):
        files = run_duration_set(duration, repetitions, cool_down_between_measurements)
        all_files.extend(files)
        
        # Cool down between different duration sets (except after the last one)
        if i < len(durations) - 1:
            print(f"\nCooling down for {cool_down_between_sets} seconds before next duration set...")
            time.sleep(cool_down_between_sets)
    
    print("\n=== Collection Complete ===")
    print(f"Total files created: {len(all_files)}")
    
    # Create overall summary file
    with open("power_measurements_summary.txt", "w") as f:
        f.write("# PC Power Statistics Summary\n")
        f.write(f"# Collection completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Durations measured: {durations} seconds\n")
        f.write(f"# Repetitions per duration: {repetitions}\n\n")
        
        f.write("## Files created:\n")
        for duration in durations:
            f.write(f"\n# {duration} seconds duration:\n")
            for file in all_files:
                if file.startswith(f"{duration}_sec_"):
                    f.write(f"- {file}\n")
    
    print(f"Summary saved to power_measurements_summary.txt")

if __name__ == "__main__":
    main()