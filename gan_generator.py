import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import array_to_img

np.random.seed(42)


class GANImageGenerator:

    def __init__(self, model_path):
        print("Loading GAN generator model...")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.generator = load_model(model_path)
        self.latent_dim = self.generator.input_shape[1]
        self.img_size = self.generator.output_shape[1]

        print(f"\n[SUCCESS] Generator loaded!")
        print(f"  - Input dimension: {self.latent_dim}")
        print(f"  - Output size: {self.img_size}x{self.img_size}")
        print(f"  - Model parameters: {self.generator.count_params():,}")

    def generate_images(self, num_images, output_dir='gan-generated',
                        batch_size=32, upscale_to=224, prefix='gan_'):

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        print(f"\n{'=' * 70}")
        print("GENERATING SYNTHETIC IMAGES")
        print(f"{'=' * 70}")
        print(f"Number of images: {num_images}")
        print(f"Output directory: {output_dir}")
        print(f"Batch size: {batch_size}")
        if upscale_to:
            print(f"Upscaling to: {upscale_to}x{upscale_to}")

        num_batches = (num_images + batch_size - 1) // batch_size
        generated_count = 0

        for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
            current_batch_size = min(batch_size, num_images - generated_count)

            noise = np.random.normal(0, 1, (current_batch_size, self.latent_dim))

            generated_imgs = self.generator.predict(noise, verbose=0)

            generated_imgs = 0.5 * generated_imgs + 0.5
            generated_imgs = np.clip(generated_imgs, 0, 1)
            generated_imgs = (generated_imgs * 255).astype(np.uint8)

            for i in range(current_batch_size):
                img = generated_imgs[i, :, :, 0]

                pil_img = array_to_img(img.reshape(self.img_size, self.img_size, 1))

                if upscale_to and upscale_to != self.img_size:
                    pil_img = pil_img.resize((upscale_to, upscale_to))

                filename = output_path / f'{prefix}{generated_count:04d}.jpeg'
                pil_img.save(filename, quality=95)
                generated_count += 1

        print(f"\n[SUCCESS] Generated {generated_count} images!")
        print(f"Images saved to: {output_dir}/")

        return output_path

    def generate_preview_grid(self, num_images=16, save_path='gan_preview.png'):

        import matplotlib.pyplot as plt

        print(f"\nGenerating preview grid with {num_images} images...")

        noise = np.random.normal(0, 1, (num_images, self.latent_dim))
        imgs = self.generator.predict(noise, verbose=0)

        imgs = 0.5 * imgs + 0.5
        imgs = np.clip(imgs, 0, 1)

        grid_size = int(np.sqrt(num_images))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))

        for i, ax in enumerate(axes.flat):
            if i < len(imgs):
                ax.imshow(imgs[i, :, :, 0], cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

        plt.suptitle('GAN Generated Samples', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Preview grid saved as: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate images using trained GAN')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to generator model (.h5 file)')
    parser.add_argument('--num_images', type=int, default=2534,
                        help='Number of images to generate')
    parser.add_argument('--output_dir', type=str, default='gan-generated',
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Generation batch size')
    parser.add_argument('--upscale', type=int, default=224,
                        help='Upscale images to this size (0 to disable)')
    parser.add_argument('--prefix', type=str, default='gan_',
                        help='Prefix for generated filenames')
    parser.add_argument('--preview_only', action='store_true',
                        help='Only generate preview grid, do not generate all images')

    args = parser.parse_args()

    generator = GANImageGenerator(model_path=args.model)

    if args.preview_only:
        print("\nGenerating preview only...")
        generator.generate_preview_grid(num_images=16, save_path='gan_preview.png')
        print("\n[DONE] Check 'gan_preview.png'")
    else:
        generator.generate_preview_grid(num_images=16, save_path='gan_preview.png')

        output_path = generator.generate_images(
            num_images=args.num_images,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            upscale_to=args.upscale if args.upscale > 0 else None,
            prefix=args.prefix
        )

        print("\n" + "=" * 70)
        print("GENERATION COMPLETE!")
        print("=" * 70)
        print(f"\nGenerated files: {output_path}")
        print(f"Number of images: {args.num_images}")
        print(f"Preview: gan_preview.png")

        print("\nNext steps:")
        print("  1. Review generated images")
        print("  2. Copy to training folder:")
        print(f"     cp {args.output_dir}/*.jpeg chest_xray/train_balanced/NORMAL/")


if __name__ == "__main__":
    main()
