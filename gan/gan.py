from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

class WGAN_GP:

    def __init__(self, img_size=64, latent_dim=128):
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.img_shape = (img_size, img_size, 1)
        self.gp_weight = 10.0

        self.generator = self.build_generator()
        self.critic = self.build_critic()

        self.generator_optimizer = Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.9)
        self.critic_optimizer = Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.9)

        print(f"\n{'='*70}")
        print("WGAN-GP ARCHITECTURE")
        print(f"{'='*70}")
        print(f"Image size: {img_size}x{img_size}")
        print(f"Latent dimension: {latent_dim}")
        print(f"Generator parameters: {self.generator.count_params():,}")
        print(f"Critic parameters: {self.critic.count_params():,}")

    def build_generator(self):
        noise_input = layers.Input(shape=(self.latent_dim,))

        x = layers.Dense(8 * 8 * 256)(noise_input)
        x = layers.Reshape((8, 8, 256))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = self.upsample_block(x, 256)  
        x = self.upsample_block(x, 128)  
        x = self.upsample_block(x, 64)   

        x = layers.Conv2D(1, kernel_size=7, padding='same', activation='tanh')(x)

        return models.Model(noise_input, x, name='Generator')

    def upsample_block(self, x, filters):
        x = layers.UpSampling2D(size=2)(x)
        x = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    def build_critic(self):
        img_input = layers.Input(shape=self.img_shape)

        x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same')(img_input)
        x = layers.LeakyReLU(0.2)(x)

        x = self.critic_block(x, 128)
        x = self.critic_block(x, 256)
        x = self.critic_block(x, 512)

        x = layers.Flatten()(x)
        x = layers.Dense(1)(x)  

        return models.Model(img_input, x, name='Critic')

    def critic_block(self, x, filters):
        x = layers.Conv2D(filters, kernel_size=4, strides=2, padding='same')(x)
        x = layers.LayerNormalization()(x)  
        x = layers.LeakyReLU(0.2)(x)
        return x

    @tf.function
    def gradient_penalty(self, real_images, fake_images):
        """Calculate gradient penalty for WGAN-GP"""
        batch_size = tf.shape(real_images)[0]

        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated = alpha * real_images + (1 - alpha) * fake_images

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.critic(interpolated, training=True)

        grads = tape.gradient(pred, interpolated)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)

        return gp

    @tf.function
    def train_step(self, real_images):
        """Single training step"""
        batch_size = tf.shape(real_images)[0]

        for _ in range(5):
            noise = tf.random.normal([batch_size, self.latent_dim])

            with tf.GradientTape() as tape:
                fake_images = self.generator(noise, training=True)

                real_output = self.critic(real_images, training=True)
                fake_output = self.critic(fake_images, training=True)

                gp = self.gradient_penalty(real_images, fake_images)

                c_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + self.gp_weight * gp

            c_grads = tape.gradient(c_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(c_grads, self.critic.trainable_variables))

        noise = tf.random.normal([batch_size, self.latent_dim])

        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            fake_output = self.critic(fake_images, training=True)
            g_loss = -tf.reduce_mean(fake_output)

        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return c_loss, g_loss

    def train(self, real_images, epochs=200, batch_size=32, save_interval=20):
        print(f"\n{'='*70}")
        print("TRAINING WGAN-GP")
        print(f"{'='*70}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Images: {len(real_images)}")
        print(f"Estimated time: ~{epochs * 0.25:.0f} minutes")

        os.makedirs('gan_samples', exist_ok=True)

        dataset = tf.data.Dataset.from_tensor_slices(real_images)
        dataset = dataset.shuffle(len(real_images)).batch(batch_size)

        c_losses = []
        g_losses = []

        for epoch in range(epochs):
            epoch_c_loss = []
            epoch_g_loss = []

            progress_bar = tqdm(dataset, desc=f"Epoch {epoch+1}/{epochs}")

            for real_batch in progress_bar:
                c_loss, g_loss = self.train_step(real_batch)

                epoch_c_loss.append(float(c_loss))
                epoch_g_loss.append(float(g_loss))

                progress_bar.set_postfix({
                    'C_loss': f'{c_loss:.3f}',
                    'G_loss': f'{g_loss:.3f}'
                })

            avg_c = np.mean(epoch_c_loss)
            avg_g = np.mean(epoch_g_loss)

            c_losses.append(avg_c)
            g_losses.append(avg_g)

            print(f"\nEpoch {epoch+1}: C_loss={avg_c:.3f} | G_loss={avg_g:.3f}")

            if (epoch + 1) % save_interval == 0 or epoch == 0:
                self.save_samples(epoch + 1)
                self.plot_losses(c_losses, g_losses, epoch + 1)

        print("\nSaving generator...")
        self.generator.save('generator_final.h5')
        print("[SUCCESS] Training complete!")

        return c_losses, g_losses

    def save_samples(self, epoch):
        noise = np.random.normal(0, 1, (16, self.latent_dim))
        generated = self.generator.predict(noise, verbose=0)
        generated = 0.5 * generated + 0.5

        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            ax.imshow(generated[i, :, :, 0], cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

        plt.suptitle(f'Epoch {epoch}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'gan_samples/epoch_{epoch:04d}.png', dpi=150)
        plt.close()

    def plot_losses(self, c_losses, g_losses, epoch):
        plt.figure(figsize=(10, 5))
        plt.plot(c_losses, label='Critic Loss', linewidth=2)
        plt.plot(g_losses, label='Generator Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('WGAN-GP Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'gan_samples/losses_{epoch:04d}.png', dpi=150)
        plt.close()
