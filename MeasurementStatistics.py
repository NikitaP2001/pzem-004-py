from datetime import datetime, timedelta
import psutil
import pandas as pd
from EnergyExtrapolator import EnergyExtrapolator
from stats import *

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
    def __init__(self, params: dict, floor_power: float = 0):
        self.data = None
        self.active = False
        self.process_dict = {}
        self.params = params
        self.floor_power = floor_power
        self.is_hepscore = params.get('hep', False)
        
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
        self.data = pd.DataFrame({
            'timedelta': pd.Series(dtype='timedelta64[ns]'),
            'taskCPU': pd.Series(dtype='float64'),
            'totalCPU': pd.Series(dtype='float64'),
            'taskPower': pd.Series(dtype='float64'),
            'taskPowerExtr': pd.Series(dtype='float64'),
            'totalPower': pd.Series(dtype='float64'),
            'Voltage': pd.Series(dtype='float64'),
            'Current': pd.Series(dtype='float64')
        })
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

    def smooth_data(self):
        if self.data is None or self.data.empty:
            raise MeasurementError("No data available for smoothing.")
        total_cells = self.data.size
        nan_count_before = self.data.isna().sum().sum()
        nan_percent_before = (nan_count_before / total_cells) * 100
        for column in self.data.columns:
            if column == 'timedelta':
                continue
            series = self.data[column]
            nan_indices = series.isna()
            if nan_indices.any():
                self.data[column] = series.interpolate(method='linear')
                self.data[column] = self.data[column].fillna(method='ffill')
                self.data[column] = self.data[column].fillna(method='bfill')
        
        if 'timedelta' in self.data.columns and self.data['timedelta'].isna().any():
            timestamps = pd.Series(
                [pd.Timedelta(seconds=i) for i in range(len(self.data))],
                index=self.data.index
            )
            self.data['timedelta'] = self.data['timedelta'].fillna(timestamps)
        
        nan_count_after = self.data.isna().sum().sum()
        nan_percent_after = (nan_count_after / total_cells) * 100
        percent_filled = nan_percent_before - nan_percent_after
        
        return percent_filled

    def end(self):
        """Stop recording measurements."""
        if not self.active:
            raise MeasurementError("Measurement is not active.")
        self.active = False

        extrapolator = EnergyExtrapolator(window_size=10, min_utilization=0.5)
        power_extr_support = 'totalPower' in self.data.columns and 'taskCPU' in self.data.columns and 'totalCPU' in self.data.columns
    
        if power_extr_support:
            # Convert to Python native types before passing to extrapolator
            total_power_values = [float(x) for x in self.data['totalPower'].tolist()]
            task_util_values = [float(x) for x in self.data['taskCPU'].tolist()]
            total_util_values = [float(x) for x in self.data['totalCPU'].tolist()]
            
            # Process data using the extrapolator
            task_power_values = extrapolator.process_data(
                total_power_values, task_util_values, total_util_values
            )
            
            # Store the calculated task power back in the DataFrame
            self.data['taskPowerExtr'] = task_power_values
            
            # Print extrapolation statistics
            stats = extrapolator.get_statistics()
            print("\n=== Energy Attribution Statistics ===")
            print(stats)

        if 'totalPower' in self.data.columns:
            self.data['taskPower'] = (self.data['totalPower'] - self.floor_power).clip(lower=0)

        self.smooth_data()
        print("Measurement ended.")
        self.__integrate_energy('totalEnergy', 'totalPower')
        self.__integrate_energy('taskEnergy', 'taskPower')
        if power_extr_support:
            self.__integrate_energy('taskEnergyExtr', 'taskPowerExtr')
    
    def process_dict_update(self, process: psutil.Process) -> psutil.Process:
        if process.pid not in self.process_dict:
            self.process_dict[process.pid] = process
            return process
        else:
            return self.process_dict[process.pid]

    def get_task_cpu(self, root_name = "containerd-shim-runc-v2"):
        taskCPU = 0
        try:
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
        return taskCPU

    def get_cpu_load(self):
        totalCPU = 0
        taskCPU = 0
        if self.is_hepscore:
            taskCPU = self.get_task_cpu() 
        elif self.params['exec']:
            name = self.params['exec'].split()[0]
            taskCPU = self.get_task_cpu(name) 
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
        idx = len(self.data)
        for col in self.data.columns:
            self.data.loc[idx, col] = measurement.get(col, None)
    
    def error_rate(self) -> float:
        if not hasattr(self, 'total_measures') or self.total_measures == 0:
            raise MeasurementError("Not a single measurement was done")
        return self.error_count / self.total_measures

    def filter_peaks(self, threshold_pct=15, min_cluster_size=3):

        if self.data is None or self.data.empty or 'totalPower' not in self.data.columns:
            raise MeasurementError("No power data available for peak filtering.")
        
        if len(self.data) < min_cluster_size + 2:
            print("Not enough data points for peak detection")
            return 0.0
        
        power_data = self.data['totalPower'].copy()
        
        kde = stats.gaussian_kde(power_data)
        x_grid = np.linspace(power_data.min(), power_data.max(), 1000)
        modal_value = x_grid[np.argmax(kde(x_grid))]
        threshold = modal_value * (1 + threshold_pct/100)
        
        peak_end_idx = 0
        for i in range(len(power_data) - min_cluster_size + 1):
            points = power_data.iloc[i:i+min_cluster_size]
            # Check if all points are within threshold of modal value
            if all(abs(p - modal_value) <= threshold for p in points):
                peak_end_idx = i
                break
        
        if peak_end_idx > 0:
            peak_end_time = self.data['timedelta'].iloc[peak_end_idx]
            peak_duration = peak_end_time.total_seconds()
            self.data = self.data.iloc[peak_end_idx:].copy()
            
            first_timestamp = self.data['timedelta'].iloc[0]
            self.data['timedelta'] = self.data['timedelta'] - first_timestamp
            
            self.data = self.data.reset_index(drop=True)
            
            if 'totalEnergy' in self.data.columns:
                self.__integrate_energy('totalEnergy', 'totalPower')
            if 'taskEnergy' in self.data.columns:
                self.__integrate_energy('taskEnergy', 'taskPower')
            
            return peak_duration
        
        return 0.0

    def getSummary(self):
        if self.data is None or self.data.empty:
            raise MeasurementError("No data available to generate summary.")
        summary = {}
        
        try:
            summary['taskPower'] = self.data['taskPower'].mean()
        except KeyError:
            summary['taskPower'] = None

        try:
            total_duration_seconds = self.data['timedelta'].iloc[-1].total_seconds()
            summary['measurement_duration'] = total_duration_seconds
            summary['totalPower'] = self.data['totalPower'].mean()
            # Add stability metrics for totalPower
            power_data = self.data['totalPower'].dropna()
            summary['totalPower_cv'] = calculate_cv(power_data)
            summary['totalPower_sem'] = calculate_sem(power_data)
            summary['totalPower_snr'] = calculate_snr(power_data)
            summary['totalPower_allan'] = calculate_allan_deviation(power_data)
            lower, upper, margin = calculate_confidence_interval(power_data)
            summary['totalPower_ci_lower'] = lower
            summary['totalPower_ci_upper'] = upper
            summary['totalPower_margin_of_error'] = margin
            
            # Add sample size and optimal duration info using actual duration
            sample_info = calculate_required_sample_size(power_data, total_duration_seconds)
            summary.update({f"totalPower_{k}": v for k, v in sample_info.items()})
            
            # Add stability evaluation relative to hardware accuracy (PZEM is 0.5%)
            stability_info = evaluate_measurement_stability(power_data, hardware_accuracy=0.5)
            summary.update({f"totalPower_{k}": v for k, v in stability_info.items()})
        except KeyError as e:
            summary['totalPower'] = None
            summary['totalPower_cv'] = None
            summary['totalPower_sem'] = None
            summary['totalPower_snr'] = None
            summary['totalPower_allan'] = None
            print(f"KeyError: {e} - Data may be missing or not recorded properly.")

        try:
            summary['taskEnergy'] = self.data['taskEnergy'].iloc[-1] if 'taskEnergy' in self.data.columns else None
        except IndexError:
            summary['taskEnergy'] = None

        try:
            summary['totalEnergy'] = self.data['totalEnergy'].iloc[-1] if 'totalEnergy' in self.data.columns else None
        except IndexError:
            summary['totalEnergy'] = None

        try:
            summary['taskEnergyExtr'] = self.data['taskEnergyExtr'].iloc[-1] if 'taskEnergyExtr' in self.data.columns else None
        except IndexError:
            summary['taskEnergyExtr'] = None
            
        # Add CPU stability metrics
        try:
            cpu_data = self.data['totalCPU'].dropna()
            summary['totalCPU_mean'] = cpu_data.mean()
            summary['totalCPU_cv'] = calculate_cv(cpu_data)
            summary['totalCPU_sem'] = calculate_sem(cpu_data)
            summary['totalCPU_snr'] = calculate_snr(cpu_data)
            summary['totalCPU_allan'] = calculate_allan_deviation(cpu_data)
        except KeyError:
            summary['totalCPU_mean'] = None
            summary['totalCPU_cv'] = None
            summary['totalCPU_sem'] = None
            summary['totalCPU_snr'] = None
            summary['totalCPU_allan'] = None

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