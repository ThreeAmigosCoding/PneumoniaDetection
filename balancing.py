import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pathlib import Path
from tqdm import tqdm
import random

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


class DataBalancer:
    
    def __init__(self, train_balanced_path, img_size=(224, 224)):
        
        self.train_path = Path(train_balanced_path)
        self.img_size = img_size
        self.class_names = ['NORMAL', 'PNEUMONIA']
        
        self.augmentation = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
    
    def count_images(self, class_name):
        
        class_path = self.train_path / class_name
        jpeg_count = len(list(class_path.glob('*.jpeg')))
        jpg_count = len(list(class_path.glob('*.jpg')))
        return jpeg_count + jpg_count
    
    def get_class_distribution(self):
        
        distribution = {}
        for class_name in self.class_names:
            distribution[class_name] = self.count_images(class_name)
        return distribution
    
    def oversample_minority_class(self, target_balance='equal'):
        
        print("\n" + "="*60)
        print("OVERSAMPLING WITH AUGMENTATION")
        print("="*60)
        
        distribution = self.get_class_distribution()
        normal_count = distribution['NORMAL']
        pneumonia_count = distribution['PNEUMONIA']
        
        print(f"\nCurrent distribution:")
        print(f"  • NORMAL: {normal_count}")
        print(f"  • PNEUMONIA: {pneumonia_count}")
        
        if target_balance == 'equal':
            target_normal = pneumonia_count
        else:
            target_normal = int(pneumonia_count * target_balance)
        
        num_to_generate = target_normal - normal_count
        
        if num_to_generate <= 0:
            print("\nDataset is already balanced!")
            return
        
        print(f"\nTarget distribution:")
        print(f"  • NORMAL: {target_normal} (need to generate {num_to_generate} images)")
        print(f"  • PNEUMONIA: {pneumonia_count}")
        
        normal_path = self.train_path / 'NORMAL'
        normal_images = list(normal_path.glob('*.jpeg')) + list(normal_path.glob('*.jpg'))
        
        print(f"\nGenerating {num_to_generate} synthetic NORMAL images...")
        
        generated_count = 0
        with tqdm(total=num_to_generate) as pbar:
            while generated_count < num_to_generate:
                source_img_path = random.choice(normal_images)
                
                img = Image.open(source_img_path)
                img = img.resize(self.img_size)
                img_array = img_to_array(img)
                img_array = img_array.reshape((1,) + img_array.shape)
                
                aug_iter = self.augmentation.flow(img_array, batch_size=1)
                augmented_img = next(aug_iter)[0].astype('uint8')
                
                save_path = normal_path / f'augmented_{generated_count:04d}.jpeg'
                augmented_pil = array_to_img(augmented_img)
                augmented_pil.save(save_path)
                
                generated_count += 1
                pbar.update(1)
        
        final_distribution = self.get_class_distribution()
        print(f"\nOversampling complete!")
        print(f"\nFinal distribution:")
        print(f"  • NORMAL: {final_distribution['NORMAL']}")
        print(f"  • PNEUMONIA: {final_distribution['PNEUMONIA']}")
        print(f"  • Balance ratio: {final_distribution['NORMAL']/final_distribution['PNEUMONIA']*100:.1f}%")
    
    def visualize_augmented_samples(self, num_samples=8):

        print("\n" + "="*60)
        print("VISUALIZING AUGMENTED SAMPLES")
        print("="*60)
        
        normal_path = self.train_path / 'NORMAL'
        
        original_images = [img for img in normal_path.glob('*.jpeg') 
                          if not img.name.startswith('augmented_')]
        original_images += [img for img in normal_path.glob('*.jpg') 
                           if not img.name.startswith('augmented_')]
        
        augmented_images = list(normal_path.glob('augmented_*.jpeg'))
        
        if not augmented_images:
            print(" No augmented images found. Run oversampling first!")
            return
        
        sample_original = random.sample(original_images, min(num_samples, len(original_images)))
        sample_augmented = random.sample(augmented_images, min(num_samples, len(augmented_images)))
        
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 5))
        
        for i in range(num_samples):
            if i < len(sample_original):
                img = cv2.imread(str(sample_original[i]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
                axes[0, i].imshow(img)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel('ORIGINAL', fontsize=12, fontweight='bold')
        
        for i in range(num_samples):
            if i < len(sample_augmented):
                img = cv2.imread(str(sample_augmented[i]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
                axes[1, i].imshow(img)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel('AUGMENTED', fontsize=12, fontweight='bold')
        
        plt.suptitle('Original vs Augmented NORMAL Images', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('original_vs_augmented.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(" Visualization saved as 'original_vs_augmented.png'")
    
    def create_balanced_generators(self, batch_size=32):
        print("\n" + "="*60)
        print("CREATING GENERATORS FROM BALANCED DATA")
        print("="*60)
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=5,
            horizontal_flip=True
        )
        
        train_generator = train_datagen.flow_from_directory(
            self.train_path.parent,
            classes=['train_balanced/NORMAL', 'train_balanced/PNEUMONIA'],
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        print(f"\nBalanced train generator created: {train_generator.samples} images")
        
        return train_generator
    
    def get_balance_statistics(self):
        distribution = self.get_class_distribution()
        total = sum(distribution.values())
        
        print("\n" + "="*60)
        print("DATASET BALANCE STATISTICS")
        print("="*60)
        
        for class_name, count in distribution.items():
            percentage = (count / total) * 100
            print(f"\n{class_name}:")
            print(f"  • Count: {count}")
            print(f"  • Percentage: {percentage:.2f}%")
        
        balance_ratio = min(distribution.values()) / max(distribution.values())
        print(f"\nBalance Ratio: {balance_ratio:.2%}")
        
        if balance_ratio >= 0.9:
            print("Dataset is well balanced!")
        elif balance_ratio >= 0.7:
            print("Dataset is moderately balanced")
        else:
            print("Dataset is imbalanced")


if __name__ == "__main__":
    
    print("Initializing DataBalancer class...")
    train_path = "chest_xray/train"
    balancer = DataBalancer(train_balanced_path=train_path, img_size=(224, 224))
    
    
    print("\nChecking current data distribution...")
    balancer.get_balance_statistics()
    
    
    print("\nStarting oversampling...")
    balancer.oversample_minority_class(
        target_balance='equal',
    )
    
    
    print("\nChecking distribution after balancing...")
    balancer.get_balance_statistics()
    
    
    print("\nVisualizing original vs augmented images...")
    balancer.visualize_augmented_samples(num_samples=8)
    
    
    print("\nCreating data generator...")
    train_generator = balancer.create_balanced_generators(batch_size=32)
    
    print("\n" + "="*60)
    print("Generator Information:")
    print(f"  • Number of batches: {len(train_generator)}")
    print(f"  • Total images: {train_generator.samples}")
    print(f"  • Class indices: {train_generator.class_indices}")
    print("="*60)
    
    
    print("\nDisplaying one batch of images...")
    x_batch, y_batch = next(train_generator)
    
    print(f"Batch shape: {x_batch.shape}")
    print(f"Labels shape: {y_batch.shape}")
    print(f"Sample labels from batch: {y_batch[:5]}")
    
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()
    
    for i in range(8):
        axes[i].imshow(x_batch[i])
        class_idx = np.argmax(y_batch[i])
        class_name = 'NORMAL' if class_idx == 0 else 'PNEUMONIA'
        axes[i].set_title(f'{class_name}')
        axes[i].axis('off')
    
    plt.suptitle('Sample Batch from Balanced Generator', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sample_batch.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ All operations completed successfully!")