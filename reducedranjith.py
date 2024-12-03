import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization,Concatenate, GlobalAveragePooling2D, Add, Activation)
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
class TripleNetFaultDetector:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None
    def residual_block(self, inputs, filters, kernel_size):
        x = Conv2D(filters, kernel_size, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        shortcut = Conv2D(filters, (1, 1), padding='same')(inputs)
        return Activation('relu')(Add()([x, shortcut]))
    def create_path(self, inputs, params):
        x = inputs
        for filters, kernel_size, pool in params:
            x = self.residual_block(x, filters, kernel_size)
            if pool: x = MaxPooling2D(pool_size=(2, 2))(x)
        return GlobalAveragePooling2D()(x)
    def create_model(self):
        inputs = Input(shape=self.input_shape)
        paths = [self.create_path(inputs, [(32, (3, 3), True), (64, (3, 3), True), (128, (3, 3), False)]),  # Spatial
            self.create_path(inputs, [(32, (5, 5), True), (64, (5, 5), True), (128, (5, 5), False)]),  # Context
            self.create_path(inputs, [(32, (1, 1), False), (64, (3, 3), True), (128, (3, 3), False)])] # Residual       
        x = Concatenate()(paths)
        for units, drop_rate in [(512, 0.4), (256, 0.3), (128, 0.2)]:
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(drop_rate)(x)
        outputs = Dense(12, activation='sigmoid')(x)
        self.model = Model(inputs, outputs)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),metrics=['accuracy'])
        return self.model
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
        callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)]
        return self.model.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    def plot_confusion_matrix(self, X_test, y_test):
        y_pred = (self.model.predict(X_test) > 0.5).astype(int)
        plt.figure(figsize=(20, 15))
        for i in range(y_test.shape[1]):
            plt.subplot(4, 3, i + 1)
            cm = confusion_matrix(y_test[:, i], y_pred[:, i])
            sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
            plt.title(f'Fault {i}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
        plt.tight_layout()
        plt.show()
def main():
    data_path = r'C:\Users\khv21\SolarPanels\mridula\mridula'
    X_train, X_test, X_val = (np.load(f'{data_path}/X_{name}_n12c.npy') for name in ['train', 'test', 'val'])
    y_train, y_test, y_val = (np.load(f'{data_path}/y_{name}_n12c.npy') for name in ['train', 'test', 'val'])
    # Initialize, train, and evaluate model
    detector = TripleNetFaultDetector(input_shape=(40, 24, 1))
    model = detector.create_model()
    model.summary()
    history = detector.train(X_train, y_train, X_val, y_val)
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    for i in range(y_test.shape[1]):
        print(f"\nFault {i}:\n", classification_report(y_test[:, i], y_pred[:, i]))
    detector.plot_confusion_matrix(X_test, y_test)
    model.save(r'C:\Users\khv21\Documents\TripleNet_fault_detector.keras')
    print("Model saved!")
if __name__ == "__main__":
    main()