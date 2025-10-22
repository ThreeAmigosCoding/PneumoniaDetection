import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pathlib import Path
import shutil

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

np.random.seed(42)
tf.random.set_seed(42)


class DataPreprocessor:

    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):

        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = ['NORMAL', 'PNEUMONIA']

    def analyze_dataset(self):

        print("=" * 60)
        print("DATASET ANALYSIS")
        print("=" * 60)

        stats = {}

        for split in ['train', 'val', 'test']:
            split_path = self.data_dir / split
            if not split_path.exists():
                print(f"Folder {split} does not exist!")
                continue

            print(f"\n {split.upper()} set:")
            print("-" * 40)

            split_stats = {'NORMAL': 0, 'PNEUMONIA': 0}
            image_dims = []

            for class_name in self.class_names:
                class_path = split_path / class_name
                if class_path.exists():
                    images = list(class_path.glob('*.jpeg')) + list(class_path.glob('*.jpg'))
                    split_stats[class_name] = len(images)

                    sample_images = np.random.choice(images, min(50, len(images)), replace=False)
                    for img_path in sample_images:
                        img = Image.open(img_path)
                        image_dims.append(img.size)

            total = sum(split_stats.values())
            print(f"  • NORMAL: {split_stats['NORMAL']} ({split_stats['NORMAL'] / total * 100:.1f}%)")
            print(f"  • PNEUMONIA: {split_stats['PNEUMONIA']} ({split_stats['PNEUMONIA'] / total * 100:.1f}%)")
            print(f"  • TOTAL: {total}")

            if image_dims:
                dims_array = np.array(image_dims)
                print(f"  • Average dimensions: {dims_array.mean(axis=0).astype(int)}")
                print(f"  • Min dimensions: {dims_array.min(axis=0)}")
                print(f"  • Max dimensions: {dims_array.max(axis=0)}")

            stats[split] = split_stats

        self._plot_class_distribution(stats)

        return stats

    def _plot_class_distribution(self, stats):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for idx, (split, split_stats) in enumerate(stats.items()):
            ax = axes[idx]
            classes = list(split_stats.keys())
            counts = list(split_stats.values())

            colors = ['#2ecc71', '#e74c3c']
            bars = ax.bar(classes, counts, color=colors, alpha=0.7, edgecolor='black')

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontweight='bold')

            ax.set_title(f'{split.upper()} set', fontsize=12, fontweight='bold')
            ax.set_ylabel('Image count')
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.show()

    def create_augmentation_generators(self):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,  # Normalizacija [0, 1]
            rotation_range=15,  # Rotacija ±15
            width_shift_range=0.1,  # Horizontalno pomeranje
            height_shift_range=0.1,  # Vertikalno pomeranje
            shear_range=0.1,  # Smicanje
            zoom_range=0.1,  # Zumiranje
            horizontal_flip=True,  # Horizontalno okretanje
            vertical_flip=False,  # Bez vertikalnog (za medicinske slike)
            fill_mode='nearest'  # Popunjavanje piksela
        )

        val_test_datagen = ImageDataGenerator(
            rescale=1. / 255
        )

        return train_datagen, val_test_datagen

    def create_data_generators(self, train_datagen, val_test_datagen):
        train_generator = train_datagen.flow_from_directory(
            self.data_dir / 'train',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )

        val_generator = val_test_datagen.flow_from_directory(
            self.data_dir / 'val',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        test_generator = val_test_datagen.flow_from_directory(
            self.data_dir / 'test',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        return train_generator, val_generator, test_generator

    def visualize_augmentation(self, num_examples=5):
        print("\n" + "=" * 60)
        print("VISUALIZATION")
        print("=" * 60)

        train_path = self.data_dir / 'train' / 'PNEUMONIA'
        sample_images = list(train_path.glob('*.jpeg'))
        if not sample_images:
            sample_images = list(train_path.glob('*.jpg'))

        if not sample_images:
            print("Image not found!")
            return

        sample_img_path = np.random.choice(sample_images)
        img = load_img(sample_img_path, target_size=self.img_size)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        fig, axes = plt.subplots(2, num_examples, figsize=(15, 6))

        for j in range(num_examples):
            axes[0, j].imshow(img)
            axes[0, j].axis('off')
            if j == num_examples // 2:
                axes[0, j].set_title('ORIEGINAL IMAGE', fontweight='bold')

        i = 0
        for batch in datagen.flow(x, batch_size=1):
            axes[1, i].imshow(batch[0].astype('uint8'))
            axes[1, i].axis('off')
            if i == num_examples // 2:
                axes[1, i].set_title('AUGMENTED VERSION', fontweight='bold')
            i += 1
            if i >= num_examples:
                break

        plt.tight_layout()
        plt.show()

    def visualize_samples(self, num_samples=8):
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 5))

        for idx, class_name in enumerate(self.class_names):
            class_path = self.data_dir / 'train' / class_name
            images = list(class_path.glob('*.jpeg')) + list(class_path.glob('*.jpg'))
            sample_images = np.random.choice(images, num_samples, replace=False)

            for i, img_path in enumerate(sample_images):
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)

                axes[idx, i].imshow(img)
                axes[idx, i].axis('off')

                if i == 0:
                    axes[idx, i].set_ylabel(class_name, fontsize=12, fontweight='bold')

        plt.suptitle('Dataset samples (224x224)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def prepare_for_oversampling(self):
        print("\n" + "=" * 60)
        print("BALANCEING")
        print("=" * 60)

        train_balanced_path = self.data_dir / 'train_balanced'

        if train_balanced_path.exists():
            print("'train_balanced' folder already exists. Skip copy.")
            return train_balanced_path

        print("Copy to original train folder...")
        shutil.copytree(self.data_dir / 'train', train_balanced_path)
        print(f"Folder created: {train_balanced_path}")

        normal_count = len(list((train_balanced_path / 'NORMAL').glob('*.jpeg')))
        normal_count += len(list((train_balanced_path / 'NORMAL').glob('*.jpg')))
        pneumonia_count = len(list((train_balanced_path / 'PNEUMONIA').glob('*.jpeg')))
        pneumonia_count += len(list((train_balanced_path / 'PNEUMONIA').glob('*.jpg')))

        print(f"\nCurrent distribution in train_balanced:")
        print(f"  • NORMAL: {normal_count}")
        print(f"  • PNEUMONIA: {pneumonia_count}")
        print(f"  • DIFF: {pneumonia_count - normal_count}")
        print(f"  • Need to generate ~{pneumonia_count - normal_count} NORMAL images for balance")

        return train_balanced_path

if __name__ == '__main__':
    preprocessor = DataPreprocessor(
        data_dir="chest_xray",
        img_size=(224, 224),
        batch_size=32
    )

    print("\n[1/6] Analyzing dataset...")
    stats = preprocessor.analyze_dataset()

    print("\n[2/6] Visualizing samples...")
    preprocessor.visualize_samples(num_samples=8)

    print("\n[3/6] Creating augmentation generators...")
    train_datagen, val_test_datagen = preprocessor.create_augmentation_generators()

    print("\n[4/6] Visualizing augmentation effects...")
    preprocessor.visualize_augmentation(num_examples=5)