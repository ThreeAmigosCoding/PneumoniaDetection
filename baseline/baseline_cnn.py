
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

np.random.seed(42)
tf.random.set_seed(42)


class BaselineCNN:

    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        self.history = None

    def build_model(self):
        model = models.Sequential(name='Baseline_CNN')

        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                padding='same', input_shape=self.input_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

        model.add(layers.Dense(self.num_classes, activation='softmax'))

        return model

    def compile_model(self, learning_rate=0.001):
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy',
                     keras.metrics.Precision(name='precision'),
                     keras.metrics.Recall(name='recall'),
                     keras.metrics.AUC(name='auc')]
        )

        print("\n" + "=" * 70)
        print("BASELINE CNN MODEL")
        print("=" * 70)
        self.model.summary()
        print(f"\nTotal parameters: {self.model.count_params():,}")

    def get_callbacks(self, checkpoint_path='checkpoints/baseline_cnn_best.h5'):
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),

            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),

            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        return callbacks

    def train(self, train_generator, val_generator, epochs=50, checkpoint_path='checkpoints/baseline_cnn_best.h5'):
        print("\n" + "=" * 70)
        print("TRAINING BASELINE CNN")
        print("=" * 70)
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {train_generator.batch_size}")

        callbacks = self.get_callbacks(checkpoint_path)

        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        print("\n[SUCCESS] Training complete!")

        return self.history

    def plot_training_history(self, save_path='results/baseline_cnn_training_history.png'):
        if self.history is None:
            print("No training history available. Train the model first.")
            return

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        axes[0, 0].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(self.history.history['precision'], label='Train Precision', linewidth=2)
        axes[1, 0].plot(self.history.history['val_precision'], label='Val Precision', linewidth=2)
        axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(self.history.history['recall'], label='Train Recall', linewidth=2)
        axes[1, 1].plot(self.history.history['val_recall'], label='Val Recall', linewidth=2)
        axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nTraining history saved to: {save_path}")

    def evaluate(self, test_generator, save_dir='results'):
        print("\n" + "=" * 70)
        print("EVALUATING BASELINE CNN")
        print("=" * 70)

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        print("Generating predictions...")
        y_pred_proba = self.model.predict(test_generator, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = test_generator.classes

        print("\n" + "-" * 70)
        print("CLASSIFICATION REPORT")
        print("-" * 70)

        class_names = list(test_generator.class_indices.keys())
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        print(report)

        with open(f'{save_dir}/baseline_cnn_classification_report.txt', 'w') as f:
            f.write("BASELINE CNN - CLASSIFICATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(report)

        self.plot_confusion_matrix(y_true, y_pred, class_names,
                                   save_path=f'{save_dir}/baseline_cnn_confusion_matrix.png')

        self.plot_roc_curve(y_true, y_pred_proba, class_names,
                            save_path=f'{save_dir}/baseline_cnn_roc_curve.png')

        self.plot_precision_recall_curve(y_true, y_pred_proba, class_names,
                                         save_path=f'{save_dir}/baseline_cnn_pr_curve.png')

        metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)

        with open(f'{save_dir}/baseline_cnn_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)

        print(f"\n[SUCCESS] Evaluation complete! Results saved to '{save_dir}/'")

        return metrics

    def plot_confusion_matrix(self, y_true, y_pred, class_names, save_path):
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Baseline CNN', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Confusion matrix saved to: {save_path}")

    def plot_roc_curve(self, y_true, y_pred_proba, class_names, save_path):
        plt.figure(figsize=(10, 8))

        y_true_onehot = keras.utils.to_categorical(y_true, num_classes=len(class_names))

        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, linewidth=2,
                     label=f'{class_name} (AUC = {roc_auc:.4f})')

        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Baseline CNN', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"ROC curve saved to: {save_path}")

    def plot_precision_recall_curve(self, y_true, y_pred_proba, class_names, save_path):
        plt.figure(figsize=(10, 8))

        y_true_onehot = keras.utils.to_categorical(y_true, num_classes=len(class_names))

        for i, class_name in enumerate(class_names):
            precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_pred_proba[:, i])

            plt.plot(recall, precision, linewidth=2, label=f'{class_name}')

        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve - Baseline CNN', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Precision-Recall curve saved to: {save_path}")

    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        y_true_onehot = keras.utils.to_categorical(y_true, num_classes=self.num_classes)

        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision_macro': float(precision_score(y_true, y_pred, average='macro')),
            'recall_macro': float(recall_score(y_true, y_pred, average='macro')),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
            'auc_macro': float(roc_auc_score(y_true_onehot, y_pred_proba, average='macro')),
            'precision_weighted': float(precision_score(y_true, y_pred, average='weighted')),
            'recall_weighted': float(recall_score(y_true, y_pred, average='weighted')),
            'f1_weighted': float(f1_score(y_true, y_pred, average='weighted')),
        }

        print("\n" + "-" * 70)
        print("METRICS SUMMARY")
        print("-" * 70)
        print(f"Accuracy:          {metrics['accuracy']:.4f}")
        print(f"Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (macro):    {metrics['recall_macro']:.4f}")
        print(f"F1-Score (macro):  {metrics['f1_macro']:.4f}")
        print(f"AUC (macro):       {metrics['auc_macro']:.4f}")

        return metrics

    def save_model(self, filepath='models/baseline_cnn_final.h5'):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        print(f"\nModel saved to: {filepath}")

    def load_model(self, filepath):
        self.model = keras.models.load_model(filepath)
        print(f"\nModel loaded from: {filepath}")