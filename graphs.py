import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

class TripleNetFaultDetector:
    
    def __init__(self, model_path):
        # Load the trained model from file
        self.model = load_model(model_path)
    
    def plot_accuracy_loss(self, history):
        # Plot accuracy and loss
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Accuracy
        ax1.plot(history['accuracy'], label='Train Accuracy')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        # Loss
        ax2.plot(history['loss'], label='Train Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()

        plt.tight_layout()
        plt.show()

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
    # Load datasets (make sure these are loaded after training)
    X_test = np.load(r'C:\Users\khv21\SolarPanels\mridula\mridula\X_test_n12c.npy')
    y_test = np.load(r'C:\Users\khv21\SolarPanels\mridula\mridula\y_test_n12c.npy')

    # Load the trained model and its history
    detector = TripleNetFaultDetector(model_path=r'C:\Users\khv21\Documents\TripleNet_fault_detector.keras')

    # If you have a saved history from the training (e.g., from the `fit` method), you can directly load it
    # For example, history = np.load('history.npy', allow_pickle=True).item()
    # For now, we assume 'history' is a dictionary-like object that contains 'accuracy' and 'loss'
    # Example: history = {'accuracy': [...], 'val_accuracy': [...], 'loss': [...], 'val_loss': [...]}

    # For demonstration, assuming you have the history object saved.
    history = {
        'accuracy': [0.80, 0.85, 0.88],  # Example accuracy history
        'val_accuracy': [0.75, 0.80, 0.82],
        'loss': [0.45, 0.40, 0.35],  # Example loss history
        'val_loss': [0.50, 0.45, 0.42]
    }

    # Plot accuracy and loss
    detector.plot_accuracy_loss(history)

    # Plot confusion matrices
    detector.plot_confusion_matrix(X_test, y_test)

if __name__ == "__main__":
    main()
