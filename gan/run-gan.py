import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from gan.gan import WGAN_GP


def load_images(data_path, class_name='PNEUMONIA', max_images=1341, img_size=64):
    print(f"\n{'='*70}")
    print(f"LOADING {class_name} IMAGES")
    print(f"{'='*70}")

    class_path = Path(data_path) / class_name
    image_paths = list(class_path.glob('*.jpeg')) + list(class_path.glob('*.jpg'))

    print(f"Found {len(image_paths)} images")

    if len(image_paths) > max_images:
        image_paths = np.random.choice(image_paths, max_images, replace=False)

    images = []

    for img_path in tqdm(image_paths, desc="Loading"):
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img, (img_size, img_size))

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)

            images.append(img)

        except Exception as e:
            continue

    images = np.array(images, dtype=np.float32)
    images = np.expand_dims(images, axis=-1)

    print(f"\n[SUCCESS] Loaded {len(images)} images")
    print(f"  Shape: {images.shape}")
    print(f"  Range: [{images.min():.1f}, {images.max():.1f}]")

    images = (images - 127.5) / 127.5
    print(f"  After normalization: [{images.min():.3f}, {images.max():.3f}]")

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i, :, :, 0], cmap='gray', vmin=-1, vmax=1)
            ax.axis('off')
    plt.suptitle('Sample Images (with CLAHE enhancement)', fontsize=14)
    plt.tight_layout()
    plt.show()

    return images

def compare():
    noise = np.random.normal(0, 1, (16, 128))
    generated = wgan.generator.predict(noise)
    generated = 0.5 * generated + 0.5

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated[i, :, :, 0], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    plt.suptitle('Final Generated Samples', fontsize=16)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))

    for i in range(8):
        idx = np.random.randint(0, len(real_images))
        axes[0, i].imshow(0.5 * real_images[idx, :, :, 0] + 0.5, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('REAL', fontsize=12, fontweight='bold')

        axes[1, i].imshow(generated[i, :, :, 0], cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('GENERATED', fontsize=12, fontweight='bold')

    plt.suptitle('Real vs Generated', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("\n[DONE] Check the images above!")

DATA_PATH = "../chest_xray/train"
IMG_SIZE = 64
CLASS_NAME = "PNEUMONIA"
real_images = load_images(DATA_PATH, CLASS_NAME, max_images=1500, img_size=IMG_SIZE)

wgan = WGAN_GP(img_size=IMG_SIZE, latent_dim=128)
EPOCHS = 200
BATCH_SIZE = 32

c_losses, g_losses = wgan.train(
    real_images=real_images,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    save_interval=20
)

compare()


print(f"\n[READY] Data loaded. {len(real_images)} images ready for training.")