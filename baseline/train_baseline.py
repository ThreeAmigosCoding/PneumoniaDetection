import matplotlib
matplotlib.use('Agg')

from preprocessing import DataPreprocessor
from balancing import DataBalancer
from baseline_cnn import BaselineCNN


def main():

    # ========================================================================
    # CONFIG
    # ========================================================================

    DATA_DIR = "dummy_chest_xray"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    TARGET_BALANCE = 'equal'
    EPOCHS = 50
    
    print("\n" + "="*70)
    print(" " * 15 + "PNEUMONIA DETECTION - DATA PIPELINE")
    print("="*70)
    
    # =========================================================================
    # STEP 1: PREPROCESSING
    # =========================================================================

    print("\n" + "PHASE 1: PREPROCESSING")
    print("-"*70)
    
    preprocessor = DataPreprocessor(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    print("\n[1/6] Analyzing dataset...")
    stats = preprocessor.analyze_dataset()
    
    print("\n[2/6] Visualizing samples...")
    preprocessor.visualize_samples(num_samples=8)
    
    print("\n[3/6] Creating augmentation generators...")
    train_datagen, val_test_datagen = preprocessor.create_augmentation_generators()
    
    print("\n[4/6] Visualizing augmentation effects...")
    preprocessor.visualize_augmentation(num_examples=5)
    
    print("\n[5/6] Creating data generators...")
    train_gen, val_gen, test_gen = preprocessor.create_data_generators(
        train_datagen, val_test_datagen
    )
    
    print(f"\nOriginal generators created:")
    print(f"   - Train: {train_gen.samples} images")
    print(f"   - Val: {val_gen.samples} images")
    print(f"   - Test: {test_gen.samples} images")
    
    print("\n[6/6] Preparing folder for balancing...")
    train_balanced_path = preprocessor.prepare_for_oversampling()
    
    # =========================================================================
    # STEP 2: BALANCING
    # =========================================================================

    print("\n\nPHASE 2: DATA BALANCING")
    print("-"*70)
    
    balancer = DataBalancer(
        train_balanced_path=train_balanced_path,
        img_size=IMG_SIZE
    )
    
    print("\n[1/4] Current distribution:")
    balancer.get_balance_statistics()
    
    print("\n[2/4] Starting oversampling...")
    balancer.oversample_minority_class(
        target_balance=TARGET_BALANCE,
        verbose=True
    )
    
    print("\n[3/4] Final distribution:")
    balancer.get_balance_statistics()
    
    print("\n[4/4] Visualizing augmented samples...")
    balancer.visualize_augmented_samples(num_samples=8)
    
    # =========================================================================
    # STEP 3: CREATE FINAL GENERATORS
    # =========================================================================

    print("\n\nPHASE 3: CREATING FINAL GENERATORS")
    print("-"*70)
    
    train_balanced_datagen = train_datagen
    
    train_balanced_gen = train_balanced_datagen.flow_from_directory(
        train_balanced_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    print(f"\nBalanced train generator: {train_balanced_gen.samples} images")
    print(f"   - Class distribution: {train_balanced_gen.classes.tolist().count(0)} NORMAL, "
          f"{train_balanced_gen.classes.tolist().count(1)} PNEUMONIA")
    
    print(f"\nVal generator: {val_gen.samples} images")
    print(f"Test generator: {test_gen.samples} images")

    # =========================================================================
    # PHASE 4: TRAIN BASELINE CNN
    # =========================================================================

    print("\n\nPHASE 4: TRAINING BASELINE CNN MODEL")
    print("-" * 70)

    print("\n[1/4] Initializing Baseline CNN...")
    baseline = BaselineCNN(input_shape=(224, 224, 3), num_classes=2)
    baseline.compile_model(learning_rate=0.001)

    print("\n[2/4] Training model...")
    print(f"   - Epochs: {EPOCHS}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Training samples: {train_balanced_gen.samples}")
    print(f"   - Validation samples: {val_gen.samples}")

    history = baseline.train(
        train_generator=train_balanced_gen,
        val_generator=val_gen,
        epochs=EPOCHS,
        checkpoint_path='checkpoints/baseline_cnn_best.h5'
    )

    print("\n[3/4] Plotting training history...")
    baseline.plot_training_history(save_path='results/baseline_cnn_training_history.png')

    print("\n[4/4] Evaluating on test set...")
    metrics = baseline.evaluate(
        test_generator=test_gen,
        save_dir='results'
    )

    baseline.save_model('models/baseline_cnn_final.h5')

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n\n" + "=" * 70)
    print(" " * 25 + "PIPELINE COMPLETE!")
    print("=" * 70)

    print("\nFINAL SUMMARY:")
    print("-" * 70)
    print(f"Training set (balanced):   {train_balanced_gen.samples} images")
    print(f"Validation set:            {val_gen.samples} images")
    print(f"Test set:                  {test_gen.samples} images")

    print("\nBASELINE CNN RESULTS:")
    print("-" * 70)
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"F1-Score (macro):  {metrics['f1_macro']:.4f}")
    print(f"AUC (macro):       {metrics['auc_macro']:.4f}")

    print("\nGENERATED FILES:")
    print("-" * 70)
    print("Preprocessing:")
    print("   - class_distribution.png")
    print("   - dataset_samples.png")
    print("   - augmentation_examples.png")
    print("   - original_vs_augmented.png")

    print("\nBalancing:")
    print(f"   - {train_balanced_path}/ (balanced dataset)")

    print("\nBaseline CNN:")
    print("   - results/baseline_cnn_training_history.png")
    print("   - results/baseline_cnn_confusion_matrix.png")
    print("   - results/baseline_cnn_roc_curve.png")
    print("   - results/baseline_cnn_pr_curve.png")
    print("   - results/baseline_cnn_classification_report.txt")
    print("   - results/baseline_cnn_metrics.json")
    print("   - models/baseline_cnn_final.h5")
    print("   - checkpoints/baseline_cnn_best.h5")

    print("\nFOLDERS:")
    print("-" * 70)
    print(f"   - Original train:  {DATA_DIR}/train")
    print(f"   - Balanced train:  {train_balanced_path}")
    print(f"   - Results:         results/")
    print(f"   - Models:          models/")
    print(f"   - Checkpoints:     checkpoints/")

    print("\nNEXT STEPS:")
    print("-" * 70)
    print("1. [DONE] Preprocessing")
    print("2. [DONE] Balancing")
    print("3. [DONE] Baseline CNN training")
    print("4. [TODO] Implement transfer learning models:")
    print("         - DenseNet121")
    print("         - ResNet50")
    print("         - EfficientNetB0")
    print("5. [TODO] Implement CheXNet fine-tuning")
    print("6. [TODO] Compare all models")
    print("7. [TODO] Generate final report")

    print("\n" + "=" * 70)

    return {
        'train_balanced_gen': train_balanced_gen,
        'val_gen': val_gen,
        'test_gen': test_gen,
        'stats': stats,
        'baseline_model': baseline,
        'baseline_metrics': metrics
    }


if __name__ == "__main__":
    results = main()
    print("\nGenerators saved in 'results' dictionary")
    print("   Access them with: results['train_balanced_gen'], etc.")