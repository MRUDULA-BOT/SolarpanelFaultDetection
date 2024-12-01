import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Concatenate,GlobalAveragePooling2D, Add, Activation)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ReduceLROnPlateau, EarlyStopping,ModelCheckpoint)
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
class TripleNetFaultDetector:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None
    def create_residual_block(self, inputs, filters, kernel_size):
        x = Conv2D(filters, kernel_size, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        shortcut = Conv2D(filters, (1, 1), padding='same')(inputs)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x
    def create_path(self, inputs, conv_params, pooling_size):
        x = inputs
        for filters, kernel_size in conv_params:
            x = self.create_residual_block(x, filters, kernel_size)
            x = MaxPooling2D(pooling_size)(x) if kernel_size != (1, 1) else x
        return GlobalAveragePooling2D()(x)
    def create_model(self):
        inputs = Input(shape=self.input_shape)
        # Create multiple feature extraction paths
        spatial = self.create_path(inputs, [(32, (3, 3)), (64, (3, 3)), (128, (3, 3))], (2, 2))
        context = self.create_path(inputs, [(32, (5, 5)), (64, (5, 5)), (128, (5, 5))], (2, 2))
        residual = self.create_path(inputs, [(32, (1, 1)), (64, (3, 3)), (128, (3, 3))], (2, 2))
        # Combine features
        combined = Concatenate()([spatial, context, residual])
        # Dense layers
        x = Dense(512, activation='relu')(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        # Output layer
        outputs = Dense(12, activation='sigmoid')(x)
        self.model = Model(inputs, outputs)
        # Compile the model
        optimizer = Adam(learning_rate=1e-4)
        self.model.compile(optimizer=optimizer,loss=BinaryCrossentropy(label_smoothing=0.1),metrics=['accuracy'])
        return self.model
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
        # Callbacks
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=5,min_lr=1e-6,verbose=1)
        early_stopping = EarlyStopping(monitor='val_accuracy',patience=10,restore_best_weights=True)
        model_checkpoint = ModelCheckpoint('best_model.keras',monitor='val_accuracy',save_best_only=True)
        return history
    def plot_confusion_matrix(self, X_test, y_test):
        y_pred = (self.model.predict(X_test) > 0.5).astype(int)
        plt.figure(figsize=(20, 15))
        for i in range(y_test.shape[1]):
            plt.subplot(4, 3, i + 1)
            cm = confusion_matrix(y_test[:, i], y_pred[:, i])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix for Fault {i}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
def main():
    # Load datasets
    X_train = np.load(r'C:\Users\khv21\SolarPanels\mridula\mridula\X_train_n12c.npy')
    X_test = np.load(r'C:\Users\khv21\SolarPanels\mridula\mridula\X_test_n12c.npy')
    X_val = np.load(r'C:\Users\khv21\SolarPanels\mridula\mridula\X_val_n12c.npy')
    y_train = np.load(r'C:\Users\khv21\SolarPanels\mridula\mridula\y_train_n12c.npy')
    y_test = np.load(r'C:\Users\khv21\SolarPanels\mridula\mridula\y_test_n12c.npy')
    y_val = np.load(r'C:\Users\khv21\SolarPanels\mridula\mridula\y_val_n12c.npy')
    # Initialize and build the model
    detector = TripleNetFaultDetector(input_shape=(40, 24, 1))
    model = detector.create_model()
    model.summary()
    # Train the model
    history = detector.train(X_train, y_train, X_val, y_val)
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    # Classification report
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    for i in range(y_test.shape[1]):
        print(f"\nFault {i}:\n", classification_report(y_test[:, i], y_pred[:, i]))
    # Plot confusion matrices
    detector.plot_confusion_matrix(X_test, y_test)
    model.save(r'C:\Users\khv21\Documents\TripleNet_fault_detector.keras')
    print("Model saved!")
if __name__ == "__main__":
    main()