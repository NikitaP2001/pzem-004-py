import matplotlib.pyplot as plt
from MeasurementStatistics import MeasurementError

class MeasurementPlotter:
    """
    A class to plot measurement statistics.

    Attributes:
    - statistics (MeasurementStatistics): The instance containing recorded data.
    """
    UNITS = {
        "taskEnergy": "Task Energy (J)",
        "taskPower": "Task Power (W)",
        "taskPowerExtr": "Task Power Extr (W)",
        "taskCPU": "Task CPU (%)",
        "totalEnergy": "Total Energy (J)",
        "taskEnergyExtr": "Total Energy Extr (J)",
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
        fig, axs = plt.subplots(4, 2, figsize=(12, 10), sharex=True)  # Changed to 4 rows
        fig.suptitle('Measurement Statistics Over Time', fontsize=16)

        # Data for plotting
        columns = ['taskEnergy', 'taskPower', 'taskCPU', 'taskEnergyExtr', 'taskPowerExtr',
            'totalEnergy', 'totalPower', 'totalCPU']
        ax_positions = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1),
                   (0, 1), (1, 1), (2, 1)]

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