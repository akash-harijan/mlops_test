import tensorflow as tf
from PIL import Image
import numpy as np

def save_mnist_image_as_jpg(image_index=0, save_path='mnist_image.jpg'):
    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Select an image by index
    image = train_images[image_index]  # Change index to select different images

    # Normalize and convert the image to uint8
    image = ((image / 255.0) * 255).astype(np.uint8)

    # Convert numpy array to a PIL image
    pil_image = Image.fromarray(image)

    # Save the image
    pil_image.save(save_path, format='JPEG')

    print(f"Image saved to {save_path}")

# Example usage
save_mnist_image_as_jpg(image_index=0, save_path='mnist_image.jpg')
