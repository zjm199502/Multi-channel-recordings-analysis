
# This .py will do:
#   1. load whole spike raster from curated phy2 data and excludes group 'noise', and save as: data_transposed.csv
#   2. Calculate the average firing rate across the entire recording, and save as: neuron_Freq_entireR.xlsx


import spikeinterface.extractors as se
import csv
from pathlib import Path
import pandas as pd
import numpy as np

base_folder = Path("/Users/zhangjinming/Documents/Open Ephys/CSDS/2_2023-11-25_13-31-27")
sample_freq        = 30000 # Hz
timestamps         = np.load(base_folder / 'Record Node 119/experiment1/recording1/continuous'
                                           '/Rhythm_FPGA-100.0/timestamps.npy')
total_recording_time = len(timestamps) / sample_freq  # seconds
print(total_recording_time)

# Part 1
# Reading spike train data and exporting to CSV
sorting_SC2 = se.PhySortingExtractor(base_folder / 'phy_SC2/', exclude_cluster_groups=['noise'])

csv_file_path = base_folder / 'data.csv'

with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    unitlist = list(sorting_SC2.get_unit_ids())
    for unit_id in unitlist:
        spike_train = sorting_SC2.get_unit_spike_train(unit_id=unit_id, return_times=True)
        csv_writer.writerow(spike_train)

print(f"Spike timepoints exported to {csv_file_path}")

# Reading CSV, transposing data, and exporting to a new CSV file using pandas
with open(csv_file_path, 'r') as file:
    csv_reader = csv.reader(file)
    data = [row for row in csv_reader]

df = pd.DataFrame(data)
df_transposed = df.transpose()

df_transposed.to_csv(base_folder / 'data_transposed.csv', header=unitlist, index=False)
print(f"Spike_rast_transposed exported to {base_folder / 'data_transposed.csv'}")

# Part 2
# Create an Excel writer for average frequency across the entire recording
excel_writer_neuron_Freq_entireR = pd.ExcelWriter(base_folder / 'neuron_Freq_entireR.xlsx', engine='xlsxwriter')
file_path_spike_raster = base_folder / 'data_transposed.csv'

# Load spike raster data from a CSV file (assuming each column represents a neuron)
df_spikes_raw = pd.read_csv(file_path_spike_raster)

# Calculate average frequency for each neuron across the entire recording
neuron_avg_freq_entireR = []

for neuron_col in df_spikes_raw.columns:
    # Extract spikes for the current neuron
    spikes_neuron = df_spikes_raw[neuron_col].values
    # Calculate average frequency for the entire recording
    avg_freq_entireR = len(np.unique(spikes_neuron))/ total_recording_time  # Assuming total_recording_time is defined
    # Append the result to the list
    neuron_avg_freq_entireR.append({'Neuron': neuron_col, 'Average_Frequency_EntireR': avg_freq_entireR})

# Convert the result to a DataFrame
neuron_avg_freq_entireR_df = pd.DataFrame(neuron_avg_freq_entireR)
print(neuron_avg_freq_entireR_df)
# Save the DataFrame to Excel
neuron_avg_freq_entireR_df.to_excel(excel_writer_neuron_Freq_entireR, index=False, sheet_name='Neuron_Avg_Freq_EntireR')
excel_writer_neuron_Freq_entireR._save()

print(f"neuron_Freq_entireR.xlsx exported to {base_folder}")

