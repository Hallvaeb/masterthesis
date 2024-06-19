import logging
import datetime
import time

class LoggerHallvard:
    _instance = None  # For singleton

    def __new__(cls, start_time=0, log_name="LoggerHallvard", log_directory="/home/pi/logs_hallvard/", hm_id=40):
        if cls._instance is None:
            cls._instance = super(LoggerHallvard, cls).__new__(cls)
            cls._instance.setup(start_time, log_name, log_directory, hm_id)
        return cls._instance

    def setup(self, start_time, log_name, log_directory, hm_id):

        self.start_time = start_time
        self.logger = logging.getLogger(log_name)
        self.log_directory = log_directory
        self.hm_id = hm_id
        self.logger.setLevel(logging.INFO)

        log_filename = datetime.datetime.now().strftime(f"{log_directory}hm{hm_id}-%d-%m-%Y.txt")

        file_handler = logging.FileHandler(log_filename, mode='a')

        # Define the format for log messages
        formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%H:%M:%S')

        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        if not self.logger.handlers:  # To avoid adding handlers multiple times if the logger is reused
            self.logger.addHandler(file_handler)

    def format_message(self, message):
        run_time = "{:05.2f}s".format(time.time() - self.start_time)
        formatted_message = f"{run_time} {message}"
        return formatted_message
    
    def print(self, message):
        print(f"LoggerHallvard for {self.hm_id}: {self.format_message(message)}")

    def write(self, message, write_to_log=True):
        if write_to_log: self.logger.info(self.format_message(message))
        print(f"Message logged to {self.log_directory}:")
        self.print(message)