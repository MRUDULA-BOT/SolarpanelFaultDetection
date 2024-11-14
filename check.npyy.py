import os
import numpy as np

folder_path = r"D:\whirl2\mridula\mridula"
files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

for file in files:
    data = np.load(os.path.join(folder_path, file))
    print(f"Data from {file}:")
    print(data.shape)
