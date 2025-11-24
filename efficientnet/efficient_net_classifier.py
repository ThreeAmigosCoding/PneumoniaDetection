import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

np.random.seed(42)
tf.random.set_seed(42)


class EfficientNetB0Transfer:
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=2, freeze_base=True):

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.freeze_base = freeze_base
        self.model = self.build_model()
        self.history = None
        
    def build_model(self):
        
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        if self.freeze_base:
            base_model.trainable = False
            print("\n[INFO] Base EfficientNetB0 layers frozen (will unfreeze later for fine-tuning)")
        else:
            base_model.trainable = True
            print("\n[INFO] Base EfficientNetB0 layers unfrozen (training from start)")
        
        inputs = keras.Input(shape=self.input_shape)
        
        x = keras.applications.efficientnet.preprocess_input(inputs)
        
        x = base_model(x, training=False)
        
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu', name='dense_1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu', name='dense_2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        model = models.Model(inputs, outputs, name='EfficientNetB0_Transfer')
        
        return model
    
    def compile_model(self, learning_rate=0.0001):
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy',
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')]
        )
        
        print("\n" + "="*70)
        print("EFFICIENTNETB0 TRANSFER LEARNING MODEL")
        print("="*70)
        
    def unfreeze_base_model(self, layers_to_unfreeze=50):
        print(f"\n[INFO] Unfreezing last {layers_to_unfreeze} layers for fine-tuning...")
        
        base_model = None
        
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'layers') and len(getattr(layer, 'layers', [])) > 50:
                base_model = layer
                print(f"[DEBUG] Found base model at layer {i}: {layer.name}")
                break
        
        if base_model is None:
            print("[ERROR] Could not find base EfficientNetB0 model!")
            print("[INFO] Available layers:")
            for i, layer in enumerate(self.model.layers):
                print(f"  {i}: {layer.name} ({type(layer).__name__})")
            return
        
        print(f"[INFO] Found base model: {base_model.name} with {len(base_model.layers)} layers")
        
        base_model.trainable = True
        
        frozen_count = 0
        unfrozen_count = 0
        
        for i, layer in enumerate(base_model.layers):
            if i < len(base_model.layers) - layers_to_unfreeze:
                layer.trainable = False
                frozen_count += 1
            else:
                layer.trainable = True
                unfrozen_count += 1
        
        print(f"[INFO] Frozen layers: {frozen_count}")
        print(f"[INFO] Unfrozen layers: {unfrozen_count}")

        self.model.compile(
            optimizer=Adam(learning_rate=0.000005),
            loss='categorical_crossentropy',
            metrics=['accuracy',
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')]
        )
        
        print(f"[INFO] Model recompiled with learning rate: 0.000005")
    
    def get_callbacks(self, checkpoint_path='checkpoints/efficientnetb0_best.h5'):
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
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
                min_lr=1e-8,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, train_generator, val_generator, epochs=30, 
              checkpoint_path='checkpoints/efficientnetb0_best.h5',
              fine_tune=True, fine_tune_epochs=20):
        print("\n" + "="*70)
        print("TRAINING EFFICIENTNETB0")
        print("="*70)
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")
        print(f"Initial epochs: {epochs}")
        if fine_tune:
            print(f"Fine-tuning epochs: {fine_tune_epochs}")
        
        callbacks = self.get_callbacks(checkpoint_path)
        
        print("\n" + "-"*70)
        print("PHASE 1: Training top layers (base frozen)")
        print("-"*70)
        
        history1 = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        if fine_tune:
            print("\n" + "-"*70)
            print("PHASE 2: Fine-tuning (unfreezing base layers)")
            print("-"*70)
            
            self.unfreeze_base_model(layers_to_unfreeze=50)
            
            print(f"\n[DEBUG] Phase 1 metrics: {list(history1.history.keys())}")
            
            history2 = self.model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=fine_tune_epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            print(f"\n[DEBUG] Phase 2 metrics: {list(history2.history.keys())}")
            
            common_metrics = set(history1.history.keys()) & set(history2.history.keys())
            print(f"\n[DEBUG] Common metrics: {common_metrics}")
            
            if not common_metrics:
                print("[WARNING] No common metrics! Using only Phase 1 history.")
                self.history = history1
            else:
                self.history = self._merge_histories(history1, history2)
        else:
            self.history = history1
        
        print("\n[SUCCESS] Training complete!")
        return self.history
    
    def _merge_histories(self, hist1, hist2):
        merged = keras.callbacks.History()
        merged.history = {}
        
        common_keys = set(hist1.history.keys()) & set(hist2.history.keys())
        
        print(f"\n[INFO] Merging histories with {len(common_keys)} common metrics")
        
        for key in common_keys:
            try:
                merged.history[key] = list(hist1.history[key]) + list(hist2.history[key])
                print(f"  ✓ Merged '{key}': {len(hist1.history[key])} + {len(hist2.history[key])} = {len(merged.history[key])} epochs")
            except Exception as e:
                print(f"  ✗ Failed to merge '{key}': {e}")
        
        return merged
    
    def plot_training_history(self, save_path='results/efficientnetb0_training_history.png'):
        if self.history is None:
            print("No training history available.")
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
        print("\n" + "="*70)
        print("EVALUATING EFFICIENTNETB0")
        print("="*70)
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        print("Generating predictions...")
        y_pred_proba = self.model.predict(test_generator, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = test_generator.classes
        
        print("\n" + "-"*70)
        print("CLASSIFICATION REPORT")
        print("-"*70)
        
        class_names = list(test_generator.class_indices.keys())
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        print(report)
        
        with open(f'{save_dir}/efficientnetb0_classification_report.txt', 'w') as f:
            f.write("EFFICIENTNETB0 - CLASSIFICATION REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(report)
        
        self.plot_confusion_matrix(y_true, y_pred, class_names,
                                   save_path=f'{save_dir}/efficientnetb0_confusion_matrix.png')
        self.plot_roc_curve(y_true, y_pred_proba, class_names,
                           save_path=f'{save_dir}/efficientnetb0_roc_curve.png')
        self.plot_precision_recall_curve(y_true, y_pred_proba, class_names,
                                        save_path=f'{save_dir}/efficientnetb0_pr_curve.png')
        
        metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)
        
        with open(f'{save_dir}/efficientnetb0_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\n[SUCCESS] Evaluation complete! Results saved to '{save_dir}/'")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, save_path):
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - EfficientNetB0', fontsize=14, fontweight='bold')
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
        plt.title('ROC Curve - EfficientNetB0', fontsize=14, fontweight='bold')
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
        plt.title('Precision-Recall Curve - EfficientNetB0', fontsize=14, fontweight='bold')
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
        
        print("\n" + "-"*70)
        print("METRICS SUMMARY")
        print("-"*70)
        print(f"Accuracy:          {metrics['accuracy']:.4f}")
        print(f"Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (macro):    {metrics['recall_macro']:.4f}")
        print(f"F1-Score (macro):  {metrics['f1_macro']:.4f}")
        print(f"AUC (macro):       {metrics['auc_macro']:.4f}")
        
        return metrics

    def save_model(self, filepath='models/efficientnetb0_final.h5'):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        print(f"\nModel saved to: {filepath}")
    
    def load_model(self, filepath):
        self.model = keras.models.load_model(filepath)
        print(f"\nModel loaded from: {filepath}")