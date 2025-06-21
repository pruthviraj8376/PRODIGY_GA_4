# pix2pix_script.py

import os
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow_examples.models.pix2pix import pix2pix

# Normalize function
def normalize(input_image, target_image):
    input_image = tf.cast(input_image, tf.float32) / 127.5 - 1
    target_image = tf.cast(target_image, tf.float32) / 127.5 - 1
    return input_image, target_image

if __name__ == "__main__":
    print("🔄 Script started...")
    tf.get_logger().setLevel('INFO')

    # Load dataset
    print("📥 Loading dataset...")
    dataset, _ = tfds.load('facades', with_info=True, as_supervised=True)
    train, test = dataset['train'], dataset['test']

    # Normalize images
    print("🧼 Normalizing dataset...")
    train = train.map(normalize)
    test = test.map(normalize)

    # Load pre-trained pix2pix generator
    print("📦 Loading generator model...")
    generator = pix2pix.unet_generator(output_channels=3)

    # Ensure output directory exists
    os.makedirs('results', exist_ok=True)

    # Predict and display/save result
    print("🧪 Generating prediction on one sample...")
    for inp, tar in test.take(1):
        pred = generator(tf.expand_dims(inp, 0), training=False)

        plt.figure(figsize=(15, 5))
        display_list = [inp, tar, pred[0]]
        title = ['Input', 'Target', 'Generated']

        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            plt.imshow(display_list[i] * 0.5 + 0.5)  # Convert [-1, 1] to [0, 1]
            plt.axis('off')

        output_path = 'results/output_image.png'
        plt.savefig(output_path)
        print(f"✅ Image saved at: {output_path}")
        plt.show()


