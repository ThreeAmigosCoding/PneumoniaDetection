
import os
import sys
from pathlib import Path
import json

import numpy as np
import matplotlib

matplotlib.use('Agg')
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from preprocessing import DataPreprocessor
from balancing import DataBalancer
from dense_net_classifier import DenseNet121Transfer

np.random.seed(42)
tf.random.set_seed(42)


def main():

    DATA_DIR = './dummy_chest_xray'
    PROJECT_DIR = '.'

    CHECKPOINT_DIR = f'{PROJECT_DIR}/checkpoints'
    MODEL_DIR = f'{PROJECT_DIR}/models'
    RESULTS_DIR = f'{PROJECT_DIR}/results/densenet'

    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    INITIAL_EPOCHS = 30
    FINE_TUNE_EPOCHS = 20
    LEARNING_RATE = 0.0001
    BALANCE_DATASET = True

    CHECKPOINT_PATH = f'{CHECKPOINT_DIR}/densenet_best.h5'
    MODEL_SAVE_PATH = f'{MODEL_DIR}/densenet_final.h5'

    print("=" * 70)
    print("PNEUMONIA DETECTION - DENSENET TRAINING PIPELINE")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("SYSTEM CHECK")
    print("=" * 70)
    print(f"TensorFlow version: {tf.__version__}")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU available: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("⚠ No GPU found - training will use CPU (slower)")

    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")

    if not Path(DATA_DIR).exists():
        print(f"\n❌ ERROR: Dataset directory not found at {DATA_DIR}")
        print("Please create the dataset directory with train/val/test folders!")
        return
    else:
        print(f"\n✓ Dataset directory found: {DATA_DIR}")

    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"Model:              DenseNet")
    print(f"Data directory:     {DATA_DIR}")
    print(f"Image size:         {IMG_SIZE}")
    print(f"Batch size:         {BATCH_SIZE}")
    print(f"Initial epochs:     {INITIAL_EPOCHS}")
    print(f"Fine-tune epochs:   {FINE_TUNE_EPOCHS}")
    print(f"Learning rate:      {LEARNING_RATE}")
    print(f"Balance dataset:    {BALANCE_DATASET}")
    print(f"Checkpoint path:    {CHECKPOINT_PATH}")
    print(f"Model save path:    {MODEL_SAVE_PATH}")
    print(f"Results directory:  {RESULTS_DIR}")

    print("\n" + "=" * 70)
    print("STEP 1: DATASET ANALYSIS")
    print("=" * 70)

    preprocessor = DataPreprocessor(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    stats = preprocessor.analyze_dataset()

    print("\nVisualizing dataset samples...")
    preprocessor.visualize_samples(num_samples=8)

    print("\nVisualizing augmentation effects...")
    preprocessor.visualize_augmentation(num_examples=5)

    print("\n" + "=" * 70)
    print("STEP 2: CREATING DATA GENERATORS")
    print("=" * 70)

    train_datagen, val_test_datagen = preprocessor.create_augmentation_generators()

    train_generator, val_generator, test_generator = preprocessor.create_data_generators(
        train_datagen,
        val_test_datagen
    )

    print(f"\n✓ Train generator: {train_generator.samples} images")
    print(f"✓ Validation generator: {val_generator.samples} images")
    print(f"✓ Test generator: {test_generator.samples} images")

    if BALANCE_DATASET:
        print("\n" + "=" * 70)
        print("STEP 3: DATASET BALANCING")
        print("=" * 70)

        train_balanced_path = preprocessor.prepare_for_oversampling()

        balancer = DataBalancer(
            train_balanced_path=train_balanced_path,
            img_size=IMG_SIZE
        )

        balancer.get_balance_statistics()

        print("\n⚠ This may take several minutes...")
        balancer.oversample_minority_class(target_balance='equal', verbose=True)

        print("\nVisualizing augmented samples...")
        balancer.visualize_augmented_samples(num_samples=8)

        train_datagen_balanced = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=5,
            horizontal_flip=True
        )

        train_generator = train_datagen_balanced.flow_from_directory(
            train_balanced_path,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )

        print(f"\n✓ Balanced train generator: {train_generator.samples} images")
    else:
        print("\n" + "=" * 70)
        print("STEP 3: DATASET BALANCING - SKIPPED")
        print("=" * 70)
        print("Using original imbalanced dataset")

    print("\n" + "=" * 70)
    print("STEP 4: BUILDING DENSENET MODEL")
    print("=" * 70)

    model = DenseNet121Transfer(
        input_shape=(*IMG_SIZE, 3),
        num_classes=2,
        freeze_base=True
    )

    model.compile_model(learning_rate=LEARNING_RATE)

    print("\nModel Summary:")
    model.model.summary()

    trainable_params = sum([tf.size(w).numpy() for w in model.model.trainable_weights])
    total_params = sum([tf.size(w).numpy() for w in model.model.weights])
    frozen_params = total_params - trainable_params

    print("\n" + "-" * 70)
    print("MODEL PARAMETERS")
    print("-" * 70)
    print(f"Architecture:          DenseNet")
    print(f"Trainable parameters:  {trainable_params:,}")
    print(f"Frozen parameters:     {frozen_params:,}")
    print(f"Total parameters:      {total_params:,}")
    print(f"Trainable percentage:  {trainable_params / total_params * 100:.2f}%")


    print("\n" + "=" * 70)
    print("STEP 5: TRAINING DENSENET MODEL")
    print("=" * 70)
    print("\n⚠ This will take a while... Go grab a coffee! ☕")

    try:
        history = model.train(
            train_generator=train_generator,
            val_generator=val_generator,
            epochs=INITIAL_EPOCHS,
            checkpoint_path=CHECKPOINT_PATH,
            fine_tune=True,
            fine_tune_epochs=FINE_TUNE_EPOCHS
        )

        print("\n✓ Training completed successfully!")

    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user!")
        print("Saving current model state...")
        model.save_model(filepath=f'{MODEL_DIR}/densenet_interrupted.h5')
        print("Model saved. You can resume training later.")
        return

    except Exception as e:
        print(f"\n❌ ERROR during training: {e}")
        print("Saving current model state...")
        model.save_model(filepath=f'{MODEL_DIR}/densenet_error.h5')
        raise

    print("\nGenerating training history plots...")
    model.plot_training_history(save_path=f'{RESULTS_DIR}/densenet_training_history.png')

    print("\n" + "=" * 70)
    print("STEP 6: EVALUATION ON TEST SET")
    print("=" * 70)

    metrics = model.evaluate(
        test_generator=test_generator,
        save_dir=RESULTS_DIR
    )

    print("\n" + "=" * 70)
    print("FINAL TEST METRICS - DENSENET")
    print("=" * 70)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:25s}: {metric_value:.4f}")

    print("\n" + "=" * 70)
    print("STEP 7: SAVING MODEL")
    print("=" * 70)

    model.save_model(filepath=MODEL_SAVE_PATH)

    print(f"\n✓ Model saved to: {MODEL_SAVE_PATH}")
    print(f"✓ Results saved to: {RESULTS_DIR}/")
    print(f"✓ Checkpoints saved to: {CHECKPOINT_PATH}")

    print("\n" + "=" * 70)
    print("STEP 8: ADDITIONAL ANALYSIS")
    print("=" * 70)

    print("\nLoading best model from checkpoint...")
    try:
        model.load_model(CHECKPOINT_PATH)

        print("\nRe-evaluating with best checkpoint...")
        best_metrics = model.evaluate(
            test_generator=test_generator,
            save_dir=f'{RESULTS_DIR}/best_checkpoint'
        )

        print("\n" + "=" * 70)
        print("COMPARISON: Final Model vs Best Checkpoint")
        print("=" * 70)
        print(f"{'Metric':<25} {'Final Model':<15} {'Best Checkpoint':<15} {'Difference':<15}")
        print("-" * 70)
        for key in metrics.keys():
            diff = best_metrics[key] - metrics[key]
            diff_str = f"{diff:+.4f}"
            print(f"{key:<25} {metrics[key]:<15.4f} {best_metrics[key]:<15.4f} {diff_str:<15}")

        comparison = {
            'final_model': metrics,
            'best_checkpoint': best_metrics,
            'differences': {k: best_metrics[k] - metrics[k] for k in metrics.keys()}
        }

        with open(f'{RESULTS_DIR}/model_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=4)

        print(f"\n✓ Comparison saved to: {RESULTS_DIR}/model_comparison.json")

    except Exception as e:
        print(f"\n⚠ Could not load checkpoint for comparison: {e}")
        print("Continuing with final model only...")


    print("\n" + "=" * 70)
    print("TRAINING SUMMARY - DENSENET")
    print("=" * 70)

    summary = {
        'model': 'DenseNet',
        'dataset': {
            'data_dir': DATA_DIR,
            'train_samples': train_generator.samples,
            'val_samples': val_generator.samples,
            'test_samples': test_generator.samples,
            'balanced': BALANCE_DATASET
        },
        'configuration': {
            'img_size': IMG_SIZE,
            'batch_size': BATCH_SIZE,
            'initial_epochs': INITIAL_EPOCHS,
            'fine_tune_epochs': FINE_TUNE_EPOCHS,
            'learning_rate': LEARNING_RATE
        },
        'model_info': {
            'architecture': 'DenseNet',
            'trainable_params': int(trainable_params),
            'total_params': int(total_params)
        },
        'results': {
            'test_metrics': metrics,
            'model_path': MODEL_SAVE_PATH,
            'checkpoint_path': CHECKPOINT_PATH,
            'results_dir': RESULTS_DIR
        }
    }

    with open(f'{RESULTS_DIR}/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)

    print("\nModel:")
    print(f"  • Architecture:       DenseNet")
    print(f"  • Total parameters:   {total_params:,}")
    print(f"  • Trainable params:   {trainable_params:,}")

    print("\nDataset:")
    print(f"  • Train samples:      {train_generator.samples}")
    print(f"  • Validation samples: {val_generator.samples}")
    print(f"  • Test samples:       {test_generator.samples}")
    print(f"  • Balanced:           {BALANCE_DATASET}")

    print("\nBest Test Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  • {metric_name:20s}: {metric_value:.4f}")

    print("\nFiles Created:")
    print(f"  • Model:              {MODEL_SAVE_PATH}")
    print(f"  • Checkpoint:         {CHECKPOINT_PATH}")
    print(f"  • Results:            {RESULTS_DIR}/")
    print(f"  • Summary:            {RESULTS_DIR}/training_summary.json")

    # ========================================================================
    # ZAVRŠETAK
    # ========================================================================

    print("\n" + "=" * 70)
    print("🎉 DENSENET TRAINING PIPELINE COMPLETE! 🎉")
    print("=" * 70)
    print("\n✓ All steps completed successfully!")
    print(f"\n📁 Check your results in: {RESULTS_DIR}/")
    print(f"🤖 Your trained model: {MODEL_SAVE_PATH}")
    print("\n💡 You can now use this model for predictions on new X-ray images!")
    print("\nNext step: Compare all three models (DenseNet121, ResNet50, EfficientNetB0)! 📊")


if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("STARTING DENSENET TRAINING PIPELINE")
    print("=" * 70)

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Program interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        print("\nPlease check the error message above and fix the issue.")
        sys.exit(1)