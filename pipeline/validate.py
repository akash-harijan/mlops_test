import tensorflow as tf
from data.prepare_data import prepare_mnist_data
import sqlite3

def validate_model():
    # Load data
    _, (x_val, y_val), _ = prepare_mnist_data()

    # Load model
    model = tf.keras.models.load_model("models/model.h5")

    # Evaluate model
    loss, accuracy = model.evaluate(x_val, y_val)
    print(f"Validation Accuracy: {accuracy:.2f}")

    # Store metrics in SQLite
    conn = sqlite3.connect("metrics.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS metrics (loss REAL, accuracy REAL)")
    cursor.execute("INSERT INTO metrics (loss, accuracy) VALUES (?, ?)", (loss, accuracy))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    validate_model()

