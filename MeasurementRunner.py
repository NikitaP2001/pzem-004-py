import subprocess
from time import sleep
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