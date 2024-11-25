import sys
import re
import serial
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
    validation_table = {
        "Voltage": {"suffix": "V", "low": 190, "high": 260},
        "Current": {"suffix": "A", "low": 0, "high": 100},
        "Power": {"suffix": "W", "low": 0, "high": 10000},
    }
    def __init__(self):
        self.data = None
        self.active = False
        
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

    def start(self, power_min=0):
        """Start recording measurements."""
        if self.active:
            raise MeasurementError("Measurement is already active.")
        self.data = pd.DataFrame(columns=["timedelta"] + self.REQUIRED_FIELDS)
        self.active = True
        self.error_count = 0
        self.total_measures = 0
        self.start_time = datetime.now()
        self.power_min = power_min

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
        self.data['taskPower'] = self.data['Power'].apply(lambda x: max(x - self.power_min, 0))

        print("Measurement ended.")
        self.__integrate_energy('Energy', 'Power')
        self.__integrate_energy('taskEnergy', 'taskPower')

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
        for line in lines:
            try:
                field, value = line.split(":")
                if field in self.REQUIRED_FIELDS:
                    field = field.strip()
                    valid_val = self.__validate(field, value.strip())
                    if valid_val != None:
                        measurement[field] = valid_val
            except ValueError:
                pass

        for rq_field in self.REQUIRED_FIELDS:
            if rq_field not in measurement:
                self.error_count += 1
                return

        # Append the new measurement as a row in the DataFrame
        self.data = pd.concat([self.data, pd.DataFrame([measurement])], ignore_index=True)

    def floor_power(self):
        """
        Calculate the floor power value of the system during idle operation,
        dynamically excluding the startup peak period.
        Returns:
        - float: The calculated floor power value in watts.
        """
        if self.data is None or self.data.empty:
            raise MeasurementError("No data available for calculation.")
        
        if 'Power' not in self.data.columns:
            raise MeasurementError("Power data not available in statistics.")

        power_data = self.data[['timedelta', 'Power']].copy()

        # Step 1: Identify the peak region
        global_mean = power_data['Power'].mean()
        global_std = power_data['Power'].std()

        # Identify rows that are part of the peak (more than 2 standard deviations above the mean)
        peak_condition = power_data['Power'] > (global_mean + global_std)
        peak_end_index = 0

        # Find the endpoint of the peak region (last occurrence of a peak value)
        for i in range(len(power_data)):
            if peak_condition.iloc[i]:
                peak_end_index = i
        if peak_end_index >= len(power_data) / 2:
            peak_end_index = 0 # assume no peak here

        # Exclude the peak region
        power_data = power_data.iloc[peak_end_index + 1:]
        if power_data.empty:
            raise MeasurementError("Insufficient data after excluding the peak region.")

        # Step 2: Detect and remove remaining outliers (Z-Score Method)
        mean_power = power_data['Power'].mean()
        std_power = power_data['Power'].std()
        power_data['Z-Score'] = (power_data['Power'] - mean_power) / std_power

        filtered_data = power_data[power_data['Z-Score'].abs() < 3]
        if filtered_data.empty:
            raise MeasurementError("All power values are considered outliers.")

        # Step 3: Calculate the floor power value (IQR Method)
        Q1 = filtered_data['Power'].quantile(0.25)
        Q3 = filtered_data['Power'].quantile(0.75)
        IQR = Q3 - Q1
        floor_value = Q1 - 1.5 * IQR
        # Ensure the floor value is reasonable
        floor_value = max(floor_value, 0)
        return floor_value
    
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
            summary['Current'] = self.data['Current'].mean()
        except KeyError:
            summary['Current'] = None

        try:
            summary['Voltage'] = self.data['Voltage'].mean()
        except KeyError:
            summary['Voltage'] = None

        try:
            summary['taskEnergy'] = self.data['taskEnergy'].iloc[-1] if 'taskEnergy' in self.data.columns else None
        except IndexError:
            summary['taskEnergy'] = None

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
        "Voltage": "Voltage (V)",
        "Current": "Current (A)",
        "Power": "Power (W)",
        "Energy": "Energy (J)",
        "taskPower": "Task Power (W)",
        "taskEnergy": "Task Energy (J)"
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
        columns = ['Voltage', 'Current', 'Power', 'taskPower', 'Energy', 'taskEnergy']
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

def test_arguments(params: dict, test_len: int) -> dict:
    return {
        "port": params["port"],
        "time": test_len,
        "exec": None
    }

import subprocess
from datetime import datetime, timedelta
import serial

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

def main():
    test_length = 20
    params = parse_arguments()

    stat = MeasurementStatistics()
    stat.start()

    print(f"Gathering idle system data for {test_length} seconds")
    test_params = test_arguments(params, test_length)
    reader = MeasurementRunner(test_params, stat)
    reader.run()
    stat.end()
    floor_power = stat.floor_power()

    print(f"Running main load, with idle power {floor_power}")
    reader = MeasurementRunner(params, stat)
    stat.start(floor_power)
    reader.run()
    stat.end()

    print("Error rate: ", stat.error_rate())
    print(stat.getSummary())
    plotter = MeasurementPlotter(stat)
    plotter.draw()


if __name__ == "__main__":
    main()