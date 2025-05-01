import sys
import re
import serial
import psutil
from time import sleep


import pandas as pd
from datetime import datetime, timedelta

class MeasurementError(Exception):
    """Custom exception class for MeasurementStatistics errors."""
    pass

class MeasurementStatistics:
    REQUIRED_FIELDS = [
        "Voltage", "Current", "Power"
    ]
    DATASET_FIELDS = [
        "timedelta", "taskCPU", "totalCPU", "taskPower", "totalPower"
    ]

    validation_table = {
        "Voltage": {"suffix": "V", "low": 190, "high": 260},
        "Current": {"suffix": "A", "low": 0, "high": 100},
        "Power": {"suffix": "W", "low": 0, "high": 10000},
    }
    def __init__(self, floor_power: float = 0):
        self.data = None
        self.active = False
        self.process_dict = {}
        self.floor_power = floor_power
        
    def __validate(self, field, value) -> float:
        """
        Validate and parse a measurement value.
        - field (str): The name of the field (e.g., "Voltage").
        - value (str): The value to validate (e.g., "220.40V").
        Returns:
        - float: The absolute numeric value if valid.
        - None: If the value is invalid or the field is not found.
        """
        if field not in self.validation_table:
            return None
        entry = self.validation_table[field]
        suffix = entry["suffix"]
        low_limit = entry["low"]
        high_limit = entry["high"]

        if not value.endswith(suffix):
            return None

        try:
            numeric_value = float(value.replace(suffix, "").strip())
        except ValueError:
            return None
        if low_limit <= numeric_value <= high_limit:
            return numeric_value
        return None
    
    def start(self):
        """Start recording measurements."""
        if self.active:
            raise MeasurementError("Measurement is already active.")
        self.data = pd.DataFrame(columns=self.DATASET_FIELDS)
        self.active = True
        self.error_count = 0
        self.total_measures = 0
        self.start_time = datetime.now()

    def __integrate_energy(self, energyCol: str, powerCol: str):
        self.data[energyCol] = 0.0
        for i in range(1, len(self.data)):
            # Calculate time difference (dt) in seconds
            dt = self.data.at[i, 'timedelta'] - self.data.at[i - 1, 'timedelta']
            dt = dt if isinstance(dt, (int, float)) else dt.total_seconds()
            
            # Integrate using trapezoidal rule: (P1 + P2) / 2 * dt
            power1 = self.data.at[i - 1, powerCol]
            power2 = self.data.at[i, powerCol]
            if pd.notna(power1) and pd.notna(power2):
                self.data.at[i, energyCol] = self.data.at[i - 1, energyCol] + (power1 + power2) / 2 * dt

    def end(self):
        """Stop recording measurements."""
        if not self.active:
            raise MeasurementError("Measurement is not active.")
        self.active = False
        self.data['taskPower'] = [max(power - self.floor_power, 0) for power in self.data['totalPower'] ]

        print("Measurement ended.")
        self.__integrate_energy('totalEnergy', 'totalPower')
        self.__integrate_energy('taskEnergy', 'taskPower')
    
    def process_dict_update(self, process: psutil.Process) -> psutil.Process:
        if process.pid not in self.process_dict:
            self.process_dict[process.pid] = process
            return process
        else:
            return self.process_dict[process.pid]

    def get_cpu_load(self):
        taskCPU = 0
        totalCPU = 0
        try:
            root_name = "containerd-shim-runc-v2"
            root_pid = 0 
            for process in psutil.process_iter(['pid', 'name']):
                if process.info['name'] == root_name:
                    root_pid = process.info['pid']
                    break
            process = self.process_dict_update(psutil.Process(root_pid))
            ncores = psutil.cpu_count(logical=True)
            taskCPU = process.cpu_percent(interval=0)
            for child in process.children(recursive=True):
                child = self.process_dict_update(child)
                taskCPU += child.cpu_percent(interval=0) / ncores
        except psutil.NoSuchProcess as e:
            print(e.pid, "killed before analysis")
        totalCPU = psutil.cpu_percent(interval=0)
        return taskCPU, totalCPU

    def measure(self, chunk):
        """
        Add a new chunk of data to the measurement statistics.
        Parameters:
        - chunk (str): A string containing measurement data in the format:
          "Field: Value\nField: Value\n..."
        """
        if not self.active:
            raise MeasurementError("Cannot record data. Measurement is not active. Call start() to begin.")

        timestamp = datetime.now() - self.start_time
        self.total_measures += 1
        measurement = {"timedelta": timestamp}

        lines = chunk.strip().split("\n")
        req_fields_met = len(self.REQUIRED_FIELDS)
        for line in lines:
            try:
                field, value = line.split(":")
                field = field.strip()
                # Map "Power" from input to "totalPower" in our system
               
                if field in self.REQUIRED_FIELDS:
                    name = field
                    if field == "Power":
                        name = "totalPower"
                    req_fields_met -= 1
                    valid_val = self.__validate(field, value.strip())
                    if valid_val != None:
                        measurement[name] = valid_val
            except ValueError:
                pass

        taskCPU, totalCPU = self.get_cpu_load()
        measurement["taskCPU"] = taskCPU
        measurement["totalCPU"] = totalCPU
        if req_fields_met > 0:
            self.error_count += 1
            return
                
        # Append the new measurement as a row in the DataFrame
        self.data = pd.concat([self.data, pd.DataFrame([measurement])], ignore_index=True)
    
    def error_rate(self) -> float:
        if not hasattr(self, 'total_measures') or self.total_measures == 0:
            raise MeasurementError("Not a single measurement was done")
        return self.error_count / self.total_measures

    def getSummary(self):
        if self.data is None or self.data.empty:
            raise MeasurementError("No data available to generate summary.")
        summary = {}
        try:
            summary['taskPower'] = self.data['taskPower'].mean()
        except KeyError:
            summary['taskPower'] = None

        try:
            summary['totalPower'] = self.data['totalPower'].mean()
        except KeyError:
            summary['totalPower'] = None

        try:
            summary['taskEnergy'] = self.data['taskEnergy'].iloc[-1] if 'taskEnergy' in self.data.columns else None
        except IndexError:
            summary['taskEnergy'] = None

        try:
            summary['totalEnergy'] = self.data['totalEnergy'].iloc[-1] if 'totalEnergy' in self.data.columns else None
        except IndexError:
            summary['totalEnergy'] = None

        return summary

    def get(self, field_name):
        """
        Fetch all records of a specific field.
        Parameters:
        - field_name (str): The name of the field to fetch (e.g., "Voltage").
        Returns:
        - pd.DataFrame: A DataFrame with records of the specified field and their timestamps.
        """
        if self.active:
            raise MeasurementError("Cannot fetch data while the measurement is active. Call end() first.")
        
        if self.data is None or self.data.empty:
            raise MeasurementError("No data available. Ensure data was recorded during the measurement.")
        
        if field_name not in self.data.columns:
            raise MeasurementError(f"Field '{field_name}' not found in recorded data.")
        
        # Select only the timedelta column and the requested field, dropping rows with NaN in the field
        filtered_data = self.data.loc[:, ["timedelta", field_name]].dropna(subset=[field_name])
        return filtered_data

import matplotlib.pyplot as plt

class MeasurementPlotter:
    """
    A class to plot measurement statistics.

    Attributes:
    - statistics (MeasurementStatistics): The instance containing recorded data.
    """
    UNITS = {
        "taskEnergy": "Task Energy (J)",
        "taskPower": "Task Power (W)",
        "taskCPU": "Task CPU (%)",
        "totalEnergy": "Total Energy (J)",
        "totalPower": "Total Power (W)",
        "totalCPU": "Total CPU (%)",
    }
    def __init__(self, statistics):
        """
        Initialize the MeasurementPlotter with a MeasurementStatistics instance.

        Parameters:
        - statistics (MeasurementStatistics): Instance containing recorded data.
        """
        if statistics.active:
            raise MeasurementError("Cannot initialize plotter while measurement is active. Call end() first.")

        if statistics.data is None or statistics.data.empty:
            raise MeasurementError("No data available for plotting.")

        self.statistics = statistics

    def draw(self):
        """Draw the measurements."""
        if self.statistics.data.empty:
            raise MeasurementError("No data available to plot.")
        # Get the total duration in seconds
        total_duration = self.statistics.data['timedelta'].iloc[-1].total_seconds()
        # Define a formatting function based on the total duration
        def format_time(seconds):
            if total_duration < 60:  # Less than a minute
                return f"{int(seconds)}s"
            elif total_duration < 3600:  # Less than an hour
                mins, secs = divmod(int(seconds), 60)
                return f"{mins}:{secs:02d}"
            else:  # More than an hour
                hours, remainder = divmod(int(seconds), 3600)
                mins = remainder // 60
                return f"{hours}:{mins:02d}"

        # Set up the figure with two columns of subplots
        fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
        fig.suptitle('Measurement Statistics Over Time', fontsize=16)

        # Data for plotting
        columns = ['taskEnergy', 'taskPower', 'taskCPU', 
                   'totalEnergy', 'totalPower', 'totalCPU']
        ax_positions = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]

        # Plot each measurement
        for col, (row, col_idx) in zip(columns, ax_positions):
            ax = axs[row, col_idx]
            if col in self.statistics.data.columns:
                ax.plot(self.statistics.data['timedelta'].dt.total_seconds(), 
                        self.statistics.data[col], 
                        label=self.UNITS[col])
                ax.set_ylabel(col)
                ax.legend(loc="upper right")

        # Formatting the X-axis
        for ax in axs[-1, :]:  # Only format the bottom row of subplots
            ax.set_xlabel("Time")
            ax.xaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, _: format_time(x)
            ))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    
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

    # Parse arguments
    args = parser.parse_args()

    # Validate mutually exclusive arguments
    if args.time is not None and args.exec is not None:
        parser.error("Arguments -T and -E are mutually exclusive. Specify only one.")

    # Organize arguments into a dictionary
    params = {
        "port": args.port,
        "time": args.time,
        "exec": args.exec,
    }

    return params

import subprocess
from datetime import datetime, timedelta
import serial
import os
import json
import glob

class MeasurementRunner:
    """
    A class to handle running the measurement process based on different conditions.
    Attributes:
    - params (dict): Configuration parameters for the runner (e.g., port, time, exec).
    - stat (MeasurementStatistics): Instance of MeasurementStatistics to record measurements.
    - ser (serial.Serial): Serial object for reading data from the port.
    """
    def __encode_cmd(cmd: str) -> bytes:
        return cmd.encode('ascii', errors='ignore')

    ANSWER_TMOUT = timedelta(seconds=2)
    MAX_RETRIES = 20
    START = __encode_cmd("start")
    STOP = __encode_cmd("stop")
    RESET = __encode_cmd("reset")

    def __init__(self, params, stat, buffer_max_size=1024):
        """
        Initialize the MeasurementRunner.
        Parameters:
        - params (dict): Dictionary of parameters with keys 'port', 'time', 'exec'.
        - stat (MeasurementStatistics): Instance to record measurements.
        - buffer_max_size (int): Maximum buffer size before resetting.
        """
        self.params = params
        self.stat = stat
        self.buffer_max_size = buffer_max_size
        try:
            self.ser = serial.Serial(params['port'], baudrate=115200, timeout=1)
        except serial.SerialException as e:
            raise Exception(f"Failed to connect to serial port {params['port']}: {e}")
        self.buffer = ""
        self.end_time = None
        # Set the end time if a time limit is specified
        if params['time']:
            self.end_time = datetime.now() + timedelta(seconds=params['time'])

    def __del__(self):
        """Destructor to ensure the serial port is closed."""
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()

    def __should_continue(self):
        """
        Check if the loop should continue.
        Parameters:
        - process (subprocess.Popen): The process to monitor if `exec` is specified.
        Returns:
        - bool: True if the loop should continue, False otherwise.
        """
        if self.params['exec']:
            # Continue while the process is running
            return self.process.poll() is None
        elif self.params['time']:
            # Continue until the end time is reached
            return datetime.now() < self.end_time
        else:
            # Run indefinitely
            return True
    
    def __verify(self) -> bool:
        attempts = 0
        end_time = datetime.now() + self.ANSWER_TMOUT
        while datetime.now() < end_time:
            data = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
            self.buffer += data
            if "completed" in self.buffer:
                return True
            attempts += 1
        return False

    def write_and_verify(self, cmd: bytes) -> bool:
        is_ok = False
        tries = 0
        while not is_ok and tries < self.MAX_RETRIES:
            if tries > 1:
                print(f"Failed do '{cmd.decode('utf-8', errors='ignore')}', retrying...")
            self.ser.write(cmd)
            is_ok = self.__verify()
            tries += 1
        return is_ok

    def reset(self) -> bool:
        return self.write_and_verify(self.RESET)
        
    def start(self) -> bool:
        return self.write_and_verify(self.START)

    def stop(self) -> bool:
        return self.write_and_verify(self.STOP)

    def exec(self):
        # Start the external process if `exec` is specified
        exec_str = self.params['exec']
        print("Running:", exec_str)
        self.process = subprocess.Popen(
            exec_str, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def run(self):
        """
        Run the measurement process based on the specified parameters.

        Returns:
        - MeasurementStatistics: The statistics object containing recorded measurements.
        """
        self.process = None
        try:
            if self.params['exec']:
                self.exec()

            if not self.start():
                print("\nFailed to start capturing")
                return

            while self.__should_continue():
                if self.ser.in_waiting > 0:
                    data = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
                    self.buffer += data

                    # Check if the buffer contains all required fields
                    if all(field in self.buffer for field in self.stat.REQUIRED_FIELDS):
                        self.stat.measure(self.buffer.strip())
                        self.buffer = ""

                    # Reset the buffer if it exceeds the maximum size
                    if len(self.buffer) > self.buffer_max_size:
                        print("Buffer exceeded maximum size. Resetting buffer.")
                        self.buffer = ""
                        # Wait for the next meaningful message
                        if "Voltage" in data:
                            self.buffer = data.split("Voltage", 1)[1]
                sleep(0.1)
            if not self.stop():
                print("\nFailed to stop capturing")

        except KeyboardInterrupt:
            print("\nExiting due to user interrupt.")
        finally:
            self.ser.close()
            if self.process:
                self.process.terminate()

        return self.stat

import pandas as pd

def power_get_floor(stat: MeasurementStatistics):
    totalPower = stat.get('totalPower')
    floor_power = totalPower.mean()
    return floor_power

def measure_floor_power() -> float:

    params = {
        "port": "/dev/ttyACM0",
        "time": 5,
        "exec": None,
    }
    stat = MeasurementStatistics()
    reader = MeasurementRunner(params, stat)
    stat.start()
    reader.run()
    stat.end()
    return power_get_floor(stat)

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

    print(summary)
    plotter = MeasurementPlotter(stat)
    plotter.draw()
    task_energy = int(summary['taskEnergy'])
    return { "pzemEnergy": task_energy, "hepEnergy": energy_result }

def main():
    sample_count = 1
    params = parse_arguments()
    params = {
        "port": "/dev/ttyACM0",
        "time": None,
        "exec": "sudo hep-score -v -m docker -f hepscore_short.yaml ./testdir",
    }

    floor_power = measure_floor_power()

    stat = MeasurementStatistics(floor_power)

    energy_df = pd.DataFrame(columns=["taskEnergy", "hepscoreEnergy"])
    for i in range(sample_count):
        row = measure_single_round(stat, params)
        energy_df = pd.concat([energy_df, pd.DataFrame([row])], ignore_index=True)
    
    energy_df.to_csv("atlas-gen-bmk.csv", index=False)

    # Find the HEPscore results directory


if __name__ == "__main__":
    main()