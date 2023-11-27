
# This .py will do:
#   1. load spike raster around every TTL signal, and save as: neuron_spike_raster.xlsx
#   2. Calculate the average firing rate around every TTL signal, and save as: neuron_mean_Freq.xlsx
#   3. plot histogram of firing rate around every TTL signal, and save as: neuron_histogram.xlsx
#   4. judge weather neuron firing increase, decrease or not change, and save as: pnz.xlsx


import numpy as np
import pandas as pd
from pathlib import Path

base_folder = Path("/Users/zhangjinming/Documents/Open Ephys/EZM_1_2023-11-16_10-29-12")

file_path_spike_raster = base_folder / 'data_transposed.csv'
file_path_ttl_time = base_folder / 'TTL_start.csv'

# Load spike raster data from a CSV file (assuming each column represents a neuron)
df_spikes_raw = pd.read_csv(file_path_spike_raster)
#df_spikes_raw = df_spikes_raw_full.iloc[0:]  # 表头被自动忽略了，因此此处Index应该为0，尽管第一行是神经元编号而非timestamp
print(df_spikes_raw)

# Load TTL signal times from a separate CSV file (assuming 'TTL_start.csv' has a column 'First_Position_Divided')
df_ttl = pd.read_csv(file_path_ttl_time)

# Extract TTL times as an array (assuming TTL times are in seconds)
ttl_times_seconds = df_ttl['First_Position_Divided'].values

# Set the time window around the TTL signal (500ms before and 500ms after)
time_window_before = 0.5                            # seconds
time_window_after  = 1                              # seconds
bins_size          = 0.05                           # seconds

# Create an Excel writer for histograms
excel_writer_hist = pd.ExcelWriter(base_folder / 'neuron_histograms.xlsx', engine='xlsxwriter')
# Create an Excel writer for spike raster
excel_writer_raster = pd.ExcelWriter(base_folder / 'neuron_spike_raster.xlsx', engine='xlsxwriter')
# Create an Excel writer for mean frequency
excel_writer_neuron_mean_Freq = pd.ExcelWriter(base_folder / 'neuron_mean_Freq.xlsx', engine='xlsxwriter')

# Calculate histograms and extract spike raster for each TTL event and each neuron
for i, neuron_col in enumerate(df_spikes_raw.columns):
    neuron_histograms = []
    neuron_rasters = []
    neuron_mean_Freq = []

    for ttl_time in ttl_times_seconds:
        window_start = ttl_time - time_window_before
        window_end = ttl_time + time_window_after
        bins = int((window_end - window_start) / bins_size)

        # Extract spikes within the time window for the current neuron
        spikes_in_window = df_spikes_raw[(df_spikes_raw[neuron_col] >= window_start) &
                                         (df_spikes_raw[neuron_col] <= window_end)]

        # Calculate histogram for the current neuron
        hist, bin_edges = np.histogram(spikes_in_window[neuron_col].values, bins=bins, density=False)
        neuron_histograms.append(hist)

        # Store spike raster for the current neuron
        raster_data = spikes_in_window[neuron_col].values
        raster_series = pd.Series(raster_data, name=f'Neuron_{neuron_col}_TTL_{ttl_time}')
        neuron_rasters.append(raster_series)

        # Calculate mean frequency for the current neuron
        spike_in_before = df_spikes_raw[(df_spikes_raw[neuron_col] >= window_start) &
                                        (df_spikes_raw[neuron_col] <= ttl_time)]
        spike_in_after = df_spikes_raw[(df_spikes_raw[neuron_col] >= ttl_time) &
                                       (df_spikes_raw[neuron_col] <= window_end)]
        mean_freq_before_ttl = len(spike_in_before) / time_window_before  # Hz
        mean_freq_after_ttl = len(spike_in_after) / time_window_after  # Hz
        percent_of_fre_changed = mean_freq_after_ttl / (mean_freq_before_ttl + 0.001) * 100
        neuron_mean_Freq.append([mean_freq_before_ttl, mean_freq_after_ttl, percent_of_fre_changed])

    # Convert histograms to a DataFrame
    neuron_histograms_df = pd.DataFrame(neuron_histograms, columns=[f'Bin_{i}' for i in range(1, bins + 1)])
    neuron_histograms_df.to_excel(excel_writer_hist, index=False, sheet_name=f'Neuron_{neuron_col}')

    # Convert spike rasters to a DataFrame
    neuron_rasters_df = pd.DataFrame(neuron_rasters)
    neuron_rasters_df.transpose().to_excel(excel_writer_raster, index=False, sheet_name=f'Neuron_{neuron_col}')

    # Convert mean frequency to a DataFrame
    neuron_mean_Freq_df = pd.DataFrame(neuron_mean_Freq, columns=['Before_TTL', 'After_TTL', 'Percent_Changed'])
    neuron_mean_Freq_df.to_excel(excel_writer_neuron_mean_Freq, index=False, sheet_name=f'Neuron_{neuron_col}')

# Save the Excel files
excel_writer_hist._save()
excel_writer_raster._save()
excel_writer_neuron_mean_Freq._save()

print(f"neuron_histograms.xlsx exported to {base_folder}")
print(f"neuron_spike_raster.xlsx exported to {base_folder}")
print(f"neuron_mean_Freq.xlsx exported to {base_folder}")
print(neuron_col)


# 以下代码用于计算：Percent_Changed 大于、小于或等于0的比例。
file_path_neuron_mean_Freq = base_folder / 'neuron_mean_Freq.xlsx'
df_percent_changed = pd.read_excel(file_path_neuron_mean_Freq)

# Extract TTL times as an array (assuming TTL times are in seconds)
excel_writer_pnz = pd.ExcelWriter(base_folder / 'pnz.xlsx', engine='xlsxwriter')
pnz = [] # pnz == positive vs. negative vs. zero

# Iterate over each sheet in the Excel file
for sheet_name in pd.ExcelFile(file_path_neuron_mean_Freq).sheet_names:
    df_percent_changed = pd.read_excel(file_path_neuron_mean_Freq, sheet_name=sheet_name)

    # Extract TTL times as an array (assuming TTL times are in seconds)
    caculated_percent_changed = df_percent_changed['Percent_Changed'].values

    all_mean_freq_before_ttl = np.average(df_percent_changed['Before_TTL'].values)
    all_mean_freq_after_ttl  = np.average(df_percent_changed['After_TTL'].values)
    percent_positive = np.sum(caculated_percent_changed > 120) / len(caculated_percent_changed) * 100
    percent_negative = np.sum(caculated_percent_changed < 85) / len(caculated_percent_changed) * 100

    # Determine positive_or_negative based on the conditions
    if percent_positive > percent_negative and percent_positive>=50:
        positive_or_negative = 'Positive'
    elif percent_positive < percent_negative and percent_negative>=50:
        positive_or_negative = 'Negative'
    else:
        positive_or_negative = 'Equal'

    pnz.append([all_mean_freq_before_ttl, all_mean_freq_after_ttl, percent_positive, percent_negative, positive_or_negative])

# Save the summary sheet
pnz_df_summary = pd.DataFrame(pnz, columns=['mean_fre_before', 'mean_fre_after', 'positive', 'negative', 'Direction'])
pnz_df_summary.to_excel(excel_writer_pnz, index=False, sheet_name='Summary')

excel_writer_pnz._save()