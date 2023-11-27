
# This .py will do:
#   1. load TTL signal from OpenEphys raw data (.dat, binary format),
#           transfer the TTL into a 0,1 array and save as: TTL.csv
#   2. Find where each TTL start (s) and save as: TTL_start.csv



import pandas as pd
from pathlib import Path
import Binary
import numpy as np
import matplotlib.pyplot as plt

# Define paths
base_folder = Path('/Users/zhangjinming/Documents/Open Ephys/EZM_1_2023-11-16_10-29-12')
folder = '/Users/zhangjinming/Documents/Open Ephys/EZM_1_2023-11-16_10-29-12/Record Node 101'
                # Binary.py do not support Path()
sample_freq = 30000.0  # Hz
TTL_channel = 33

# Load continuous data using openephys
data, sample_rate = Binary.Load(folder, Experiment=1, Recording=1)
# data.keys()
# data['100'].keys() # processor ID
# data['100']['0'].keys() # experiment index
# data['100']['0']['0'].shape # recording index
print(data['100']['0']['0'].shape)
# Extract TTL signal
TTL = data['100']['0']['0'][:,TTL_channel-1]
#plt.plot(TTL)
#plt.show()
# Transfer TTL signal to 0, 1
TTL_binary = np.zeros_like(TTL, dtype=int)
TTL_binary[TTL > 3e+6] = 1

plt.plot(TTL_binary)
plt.show()

# Save binary TTL signal to CSV file
pd.DataFrame(TTL_binary).to_csv(base_folder / 'TTL.csv', index=False, header=False)
print(TTL_binary)

# 读取包含数列的CSV文件
file_path = base_folder / 'TTL.csv'
df = pd.read_csv(file_path, header=None)
# 假设你的CSV文件有一个名为 'sequence' 的列，包含了0和1的数列
sequence = df.values.flatten().tolist()
# 找到每一段连续的1的第一个位置
first_positions = [i for i, x in enumerate(sequence) if x == 1 and (i == 0 or sequence[i - 1] == 0)]
# 排除第一个数值，因为它们通常指示着实验开始（由Ethovision发出）
if first_positions:
    first_positions = first_positions[1:]
# 将每一个找到的位置除以30000
first_positions_divided = [(pos + 1) / sample_freq for pos in first_positions]
# 创建一个DataFrame
result_df = pd.DataFrame({"First_Position_Divided": first_positions_divided})
# 将DataFrame保存为新的CSV文件
result_df.to_csv(base_folder / 'TTL_start.csv', index=False)
# 打印结果
print(result_df)