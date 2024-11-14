from keras.utils import to_categorical
import os
import numpy as np

folder_path = r"D:\whirl2\mridula\mridula"
files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

for file in files:
    data = np.load(os.path.join(folder_path, file))
    print(f"Data from {file}:")
    print(data.shape)

# Correct loading of y_train, y_val, and y_test
y_train = np.load(r"D:\whirl2\mridula\mridula\y_train_n12c.npy")
y_val = np.load(r"D:\whirl2\mridula\mridula\y_val_n12c.npy")
y_test = np.load(r"D:\whirl2\mridula\mridula\y_test_n12c.npy")

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=12)
y_val = to_categorical(y_val, num_classes=12)
y_test = to_categorical(y_test, num_classes=12)

# Check if they are loaded correctly
print(f'y_train shape after encoding: {y_train.shape}')
print(f'y_val shape after encoding: {y_val.shape}')
print(f'y_test shape after encoding: {y_test.shape}')

# If the labels are already in one-hot encoded form (i.e., shape is (num_samples, 12)), skip one-hot encoding
if y_train.ndim == 1 or y_train.shape[1] == 12:  # Check if they are not yet one-hot encoded
    y_train = to_categorical(y_train, num_classes=12)
    y_val = to_categorical(y_val, num_classes=12)
    y_test = to_categorical(y_test, num_classes=12)

# Print the updated shapes
print(f'y_train shape after encoding: {y_train.shape}')
print(f'y_val shape after encoding: {y_val.shape}')
print(f'y_test shape after encoding: {y_test.shape}')

