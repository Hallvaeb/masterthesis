import pandas as pd
import matplotlib.pyplot as plt
from logs_to_df import read_files_to_dataframe

opening_hours = [10, 11, 12, 13, 14, 15, 16]

# Get the detections
df = read_files_to_dataframe()

# Filter out low confidence detections
df = df[df['confidence'] > 0.5]

# Convert date and time_of_detection to datetime objects for easier analysis
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time_of_detection'], dayfirst=True)

# Extract the hour and date of detection
df['hour'] = df['datetime'].dt.hour
df['date'] = df['datetime'].dt.date

# Filter the DataFrame to include only detections within the opening hours
df_filtered = df[df['hour'].isin(opening_hours)]

# Calculate the number of unique days in the data
num_days = df_filtered['date'].nunique()

# Get the count of detections for each hour within the opening hours
hourly_counts = df_filtered['hour'].value_counts().sort_index()

# Normalize counts by the number of days
average_hourly_counts = hourly_counts / num_days

# Calculate the total number of detections within the opening hours
total_detections_within_opening_hours = hourly_counts.sum()

# Plot histogram of average detections by hour
plt.figure(figsize=(12, 8))
bars = plt.bar(average_hourly_counts.index, average_hourly_counts.values, color='blue', edgecolor='black')
# Add labels to the bars
for bar, hour in zip(bars, opening_hours):
    height = bar.get_height()
    percentage = (hourly_counts[hour]/total_detections_within_opening_hours) * 100
    label = f'{height:.0f} ({percentage:.0f}%)'
    plt.annotate(label,
				xy=(bar.get_x() + bar.get_width() / 2, height),
				xytext=(0, 3),  # 3 points vertical offset
				textcoords="offset points",
				ha='center', va='bottom')

# Set plot labels and title
plt.xlabel('Time', fontsize=14)
plt.ylabel('Average number of Persons', fontsize=14)
plt.title('Average number of Persons over the Hour', fontsize=16)
plt.xticks(opening_hours, rotation=0, labels=[f'{time}:00-{int(time)}:59' for time in opening_hours])

# Annotate total number of visitors
total_visitors_text = f'N={total_detections_within_opening_hours}'
plt.annotate(total_visitors_text,
			xy=(1, 1), xycoords='axes fraction',
			xytext=(-20, -20), textcoords='offset points',
			ha='right', va='top', fontsize=14, bbox=dict(facecolor='white', alpha=0.5))


# Save the plot
plt.savefig('./peak_hours/peak_hours.png')
# plt.show()
