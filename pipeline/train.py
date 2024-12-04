import tensorflow as tf
from data.prepare_data import prepare_mnist_data
def train_model():
    # Prepare data
    (x_train, y_train), _, _ = prepare_mnist_data()

    # Define model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train model
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

    # Save model
    model.save("models/model.h5")
    print("Model saved at models/model.h5")

if __name__ == "__main__":
    train_model()

