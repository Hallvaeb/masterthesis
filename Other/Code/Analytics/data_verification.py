"""
This will check to see if we really got detections from every 2nd minute or if there's a lot of missing data in the log file.
"""

import pandas as pd
import os
from logs_to_df import read_files_to_dataframe 

df = read_files_to_dataframe('./logs')

# Get the seconds between start_time and end_time
df['start_time'] = pd.to_datetime(df['start_time'], format='%H:%M:%S')
df['end_time'] = pd.to_datetime(df['end_time'], format='%H:%M:%S')
df['time_to_process'] = df['end_time'] - df['start_time']

# Print the rows where time_to_process is more than 70 seconds
print(df[df['time_to_process'] > '00:01:00'])

print(len(df["file_name"].unique()))